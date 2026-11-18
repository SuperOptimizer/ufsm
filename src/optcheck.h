#pragma once
/* Command-line option bookkeeping for every command (2026-11-14, mini audit opts 2-3).
   - opt_n(name): every "--name" a command compares argv against is recorded (the build wraps the literal / helper-parameter side of
     those strcmp calls); opt_p(prefix) records a prefix family (strncmp(argv[i], "--deform", 8)).
   - opt_unread(who, argc, argv): warns once per command-line --flag that no lookup has asked for so far (a typo, or an env-var
     switch passed as a flag). main() runs it for every command when the command returns; train also runs it after its setup.
   - opt_atoi / opt_atol / opt_atoll / opt_atof: atoi / atof of an option value, but a value that is not entirely a number ("2e4" for
     an integer, "288x", "abc") is an error (exit 2) instead of a silent prefix / 0. Empty or absent values stay 0. A list value
     ("0,1") gives its first element, as atoi did: the list's own parser reads the rest. */
#ifdef __cplusplus
extern "C" {
#endif
const char *opt_n(const char *name);
const char *opt_p(const char *prefix);
int opt_unread(const char *who, int argc, char **argv);
/* opt_unknown: the --flags (argv[2..]) that no command of this binary knows: not in NAMES (every "--name" literal of the sources,
   build-generated: ufsm_opt_names; a literal with printf conversions, "--aug-%s-p", is a pattern), not under one of PREFIXES (opt_p
   families), "--name=value" checked by its name. Each is named on stderr; returns how many. main() refuses them (exit 2) before any
   work unless UFSM_OPT_LAX=1 (2026-11-17: wrap-check silently ignored an unknown --bands from a stale binary; opt_unread only warns,
   and only after the command finished) */
int opt_unknown(const char *who, int argc, char **argv, const char *const *names, const char *const *prefixes);
int opt_match(const char *pat, const char *s);   /* a name pattern (printf conversions match any run) against s */
int opt_atoi(const char *name, const char *v);
long opt_atol(const char *name, const char *v);
long long opt_atoll(const char *name, const char *v);
double opt_atof(const char *name, const char *v);
#ifdef __cplusplus
}
#endif
