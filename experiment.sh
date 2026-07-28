#!/usr/bin/env bash
#
# Deploy the current working tree to the EXPERIMENT service and score it against
# production on the semantic eval.
#
# Why a separate service rather than a --no-traffic revision on production:
# deploying with --no-traffic stops the production service routing to LATEST and
# pins it to whichever revision was current. Every later deploy then creates a
# revision serving 0% while the build, the deploy and the health check all still
# report success. That silently shipped nothing three times on 2026-07-27. This
# script cannot cause that, because it never touches the production service.
#
# Usage:
#   ./experiment.sh              # deploy working tree, 2 eval runs per side
#   RUNS=3 ./experiment.sh       # more runs (the eval is noisy; see below)
#   ./experiment.sh --no-deploy  # re-score the service already deployed
#
# On noise: a single case is worth 10 points on this 10-case benchmark, and both
# routing and timeouts vary between runs. Treat a difference under ~3 points as
# nothing, and only trust a per-case change that repeats across runs.

set -euo pipefail

PROJECT=${PROJECT:-ai-discovery-platform}
REGION=${REGION:-us-central1}
PROD_SERVICE=${PROD_SERVICE:-commai-advisor}
EXP_SERVICE=${EXP_SERVICE:-commai-advisor-exp}
RUNTIME_SA=${RUNTIME_SA:-commai-advisor-runtime@ai-discovery-platform.iam.gserviceaccount.com}
RUNS=${RUNS:-2}
PYTHON=${PYTHON:-.venv-test/bin/python}

if [ "${1:-}" != "--no-deploy" ]; then
  echo "==> Deploying working tree to ${EXP_SERVICE} (production untouched)"
  gcloud run deploy "${EXP_SERVICE}" \
    --source . \
    --region "${REGION}" \
    --project "${PROJECT}" \
    --service-account "${RUNTIME_SA}" \
    --allow-unauthenticated \
    --memory 2Gi --cpu 1 --timeout 300 --concurrency 40 \
    --min-instances 0 --max-instances 3 \
    --set-secrets OPENAI_API_KEY=OPENAI_API_KEY:latest \
    --set-env-vars CHAT_MODEL=gpt-5.4-mini,EMB_MODEL=text-embedding-3-small \
    --labels purpose=experiment \
    --quiet
fi

url_of() {
  gcloud run services describe "$1" --region "${REGION}" --project "${PROJECT}" \
    --format='value(status.url)'
}

PROD_URL=$(url_of "${PROD_SERVICE}")
EXP_URL=$(url_of "${EXP_SERVICE}")

# The experiment service scales to zero, so the first request pays a cold start
# that has previously blown the eval's per-case timeout and scored a real case 0.
echo "==> Warming ${EXP_SERVICE}"
curl -fsS --max-time 180 "${EXP_URL}/health" >/dev/null || {
  echo "experiment service did not become healthy" >&2; exit 1; }

for run in $(seq 1 "${RUNS}"); do
  echo
  echo "######## run ${run}/${RUNS} — PRODUCTION ########"
  "${PYTHON}" eval_advisor_semantics.py --url "${PROD_URL}/chat" | tail -13
  echo
  echo "######## run ${run}/${RUNS} — EXPERIMENT ########"
  "${PYTHON}" eval_advisor_semantics.py --url "${EXP_URL}/chat" | tail -13
done

echo
echo "production : ${PROD_URL}"
echo "experiment : ${EXP_URL}"
echo
echo "The experiment service is not referenced by any app config and takes no"
echo "user traffic. Leave it deployed; at min-instances 0 it costs nothing idle."
