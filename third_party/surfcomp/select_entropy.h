/* Shared-model selection measures actual encoded bytes, including mask,
 * escapes, per-block stream descriptors, and the group's tables. */
typedef struct {
  se_model models[SE_NMODELS];
  se_etab etabs[SE_NMODELS];
  uint32_t counts[SE_NMODELS][SE_NTOK], dummy[SE_NMODELS][SE_NTOK];
  sfc_tokens tokens;
  uint32_t fields[SFC_TOKENS + 2];
  uint8_t ans[SFC_ANS], packet[SFC_BLOCK_BOUND];
  uint8_t trial[SFC_GROUP][SFC_BLOCK_BOUND], best[SFC_GROUP][SFC_BLOCK_BOUND];
  size_t sizes[SFC_GROUP], table_size, total;
  uint8_t tables[SE_TABLES_MAX_BYTES];
  /* ufsm: each candidate tokenized once per select_entropy call (its tokens and counts reused by every pass and packet) */
  sfc_tokens ctok[SFC_GROUP][4];
  uint32_t ccnt[SFC_GROUP][4][SE_NMODELS][SE_NTOK];
  uint8_t cok[SFC_GROUP][4];
} sfc_entropy_work;
static int cand_tokens(sfc_entropy_work *w, unsigned i, unsigned j,
                       const sfc_choice *c) {
  if (w->cok[i][j])
    return 0;
  memset(w->ccnt[i][j], 0, sizeof w->ccnt[i][j]);
  int rc = entropy_tokenize(c->bytes, c->size, &w->ctok[i][j], w->ccnt[i][j]);
  if (!rc)
    w->cok[i][j] = 1;
  return rc;
}
/* ufsm: a candidate's entropy-coded size first (entropy_packet_size: the tANS state walk alone, se_tans_size2), its bytes only for
 * the candidate a channel keeps (entropy_packet_write: the same encode as before). ent: the entropy-coded packet is the smaller. */
static int entropy_packet_size(sfc_entropy_work *w, const sfc_choice *choice,
                               unsigned ci, unsigned cj, size_t *size,
                               int *ent) {
  *size = choice->size;
  *ent = 0;
  if (choice->bytes[5] != 1)
    return 0;
  int rc = cand_tokens(w, ci, cj, choice);
  if (rc)
    return rc;
  const sfc_tokens *tk = &w->ctok[ci][cj];
  size_t an = se_tans_size2(w->etabs, tk->symbols, tk->ns, sizeof w->ans);
  if (!an)
    return 0; /* This model may not contain a competing candidate's symbols. */
  size_t at;
  if (block_prefix(choice->bytes, choice->size, &at, NULL))
    return SFC_INVALID;
  size_t cn = u32(choice->bytes + 24),
         nn = choice->size - cn + 24 + an + tk->nb;
  if (nn >= choice->size || nn > SFC_BLOCK_BOUND)
    return 0;
  *size = nn;
  *ent = 1;
  return 0;
}
static int entropy_packet_write(sfc_entropy_work *w, const sfc_choice *choice,
                                unsigned ci, unsigned cj, uint8_t *b,
                                size_t nn) {
  const sfc_tokens *tk = &w->ctok[ci][cj];
  size_t an = se_tans_encode2(w->etabs, tk->symbols, tk->ns,
                              w->fields, w->ans, sizeof w->ans);
  size_t at, cn = u32(choice->bytes + 24);
  if (!an || block_prefix(choice->bytes, choice->size, &at, NULL) ||
      nn != choice->size - cn + 24 + an + tk->nb)
    return SFC_INVALID;
  memcpy(b, choice->bytes, at);
  b[5] = 3;
  p32(b + 24, (uint32_t)(24 + an + tk->nb));
  memset(b + at, 0, 16);
  p32(b + at + 16, (uint32_t)an);
  p32(b + at + 20, (uint32_t)tk->nb);
  memcpy(b + at + 24, w->ans, an);
  memcpy(b + at + 24 + an, tk->bypass, tk->nb);
  memcpy(b + at + 24 + an + tk->nb, choice->bytes + at + cn,
         choice->size - at - cn);
  return 0;
}
static int select_entropy(sfc_candidates *const choices[], unsigned count,
                          sfc_entropy_work *w) {
  if (!count || count > SFC_GROUP)
    return SFC_INVALID;
  unsigned selected[SFC_GROUP];
  int alternatives = 0;
  memset(w->cok, 0, sizeof w->cok);
  w->total = 0;
  w->table_size = 0;
  for (unsigned i = 0; i < count; i++) {
    if (!choices[i]->count || choices[i]->preferred >= choices[i]->count)
      return SFC_INVALID;
    selected[i] = choices[i]->preferred;
    alternatives |= choices[i]->count > 1;
    sfc_choice *c = choices[i]->values + selected[i];
    w->sizes[i] = c->size;
    w->total += c->size;
    memcpy(w->best[i], c->bytes, c->size);
  }
  for (unsigned pass = 0; pass < (alternatives ? 3u : 1u); pass++) {
    memset(w->counts, 0, sizeof w->counts);
    for (unsigned i = 0; i < count; i++) {
      unsigned n = pass == 1 ? choices[i]->count : 1;
      for (unsigned j = 0; j < n; j++) {
        const sfc_choice *c =
            choices[i]->values + (pass == 1 ? j : selected[i]);
        if (c->bytes[5] == 1) {
          unsigned cj = pass == 1 ? j : selected[i];
          int rc = cand_tokens(w, i, cj, c);
          if (rc)
            return rc;
          const uint32_t *src = &w->ccnt[i][cj][0][0];
          uint32_t *dst = &w->counts[0][0];
          for (unsigned q = 0; q < SE_NMODELS * SE_NTOK; q++)
            dst[q] += src[q];
        }
      }
    }
    for (unsigned m = 0; m < SE_NMODELS; m++) {
      uint32_t total = 0;
      for (unsigned t = 0; t < SE_NTOK; t++)
        total += w->counts[m][t];
      if (!total)
        w->counts[m][0] = 1;
      if (!se_model_build_enc(w->models + m, w->counts[m]))
        return SFC_INVALID;
    }
    uint8_t tables[SE_TABLES_MAX_BYTES];
    size_t table_size = se_tables_write(w->models, tables);
    se_etabs_init(w->models, w->etabs);
    size_t total = 0, sizes[SFC_GROUP];
    int uses_tables = 0;
    unsigned win_k[SFC_GROUP];
    int win_ent[SFC_GROUP];
    for (unsigned i = 0; i < count; i++) {
      size_t minimum = SIZE_MAX, best_size = SIZE_MAX;
      double error = DBL_MAX;
      unsigned winner = selected[i];
      int went = 0;
      unsigned n = pass ? choices[i]->count : 1;
      for (unsigned j = 0; j < n; j++) {
        unsigned k = pass ? j : selected[i];
        const sfc_choice *c = choices[i]->values + k;
        size_t size;
        int ent;
        int rc = entropy_packet_size(w, c, i, k, &size, &ent);
        if (rc)
          return rc;
        if (size < minimum)
          minimum = size;
        size_t limit = minimum + minimum / 50;
        if (best_size > limit ||
            (size <= limit &&
             (c->square < error || (c->square == error && size < best_size)))) {
          best_size = size;
          error = c->square;
          winner = k;
          went = ent;
        }
      }
      selected[i] = winner;
      sizes[i] = best_size;
      total += best_size;
      win_k[i] = winner;
      win_ent[i] = went;
      uses_tables |= went; /* (the packet's byte 5 is 3 exactly when it is entropy coded) */
    }
    if (!uses_tables)
      table_size = 0;
    total += table_size;
    if (total < w->total) {
      w->total = total;
      w->table_size = table_size;
      memcpy(w->tables, tables, table_size);
      for (unsigned i = 0; i < count; i++) { /* the kept packets' bytes, with this pass's tables */
        const sfc_choice *c = choices[i]->values + win_k[i];
        w->sizes[i] = sizes[i];
        if (win_ent[i]) {
          int rc = entropy_packet_write(w, c, i, win_k[i], w->best[i], sizes[i]);
          if (rc)
            return rc;
        } else
          memcpy(w->best[i], c->bytes, sizes[i]);
      }
    }
  }
  return 0;
}
