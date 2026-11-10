#!/usr/bin/env bash
# s3ls.sh PREFIX [MAX]: list one level of the public open-data bucket (no credentials, nothing personal sent)
S=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com
curl -s "$S/?list-type=2&prefix=$1&delimiter=/&max-keys=${2:-60}" | grep -o '<Prefix>[^<]*</Prefix>\|<Key>[^<]*</Key>\|<Size>[^<]*</Size>' | sed 's/<[^>]*>//g' | paste -sd' ' | sed 's/ \([A-Za-z]\)/\n\1/g'
