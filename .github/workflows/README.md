# GitHub Actions MLOps Pipeline

This directory contains GitHub Actions workflows for continuous integration and continuous training of rental prediction models.

## Workflows

### 1. CI - Test & Validate (`ci.yml`)

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`

**What it does:**
- Tests on multiple Python versions (3.10, 3.12)
- Validates data schema with Pandera
- Runs training smoke test (quick sanity check)
- Tests preprocessing pipeline

**Purpose:** Ensure code quality and prevent breaking changes

### 2. Train Models (`train.yml`)

**Triggers:**
- **Scheduled:** Every Monday at 2 AM UTC (simulates continuous training)
- **Manual:** Via GitHub UI (Actions tab → Train Models → Run workflow)
- **Automatic:** On changes to `data/` directory

**What it does:**
1. Trains XGBoost and/or LightGBM models
2. Compares new model against production baseline
3. **Automatically promotes** if metrics improve (lower RMSE)
4. Updates `registry/production_baseline.json` if promoted
5. Saves artifacts (models, MLflow runs, reports) for 90 days
6. Creates detailed training summary in workflow output

**Inputs (manual trigger):**
- `model_type`: Which model to train (xgboost, lightgbm, both)
- `n_trials`: Number of hyperparameter tuning trials (default: 50)

## Model Promotion Logic

### Automatic Promotion Criteria

A new model is automatically promoted to production if:
1. **First run:** No baseline exists → auto-promote
2. **Improvement:** New model's test RMSE is lower than baseline → auto-promote
3. **No improvement:** Keep existing baseline

### Promotion Process

When a model is promoted:
1. `registry/production_baseline.json` is updated with new model metrics
2. Model artifacts are saved to GitHub Actions artifacts (90-day retention)
3. MLflow tracking logs all experiments
4. Changes are committed back to the repository
5. Workflow summary shows promotion decision

## File Structure

```
.github/workflows/
├── README.md           # This file
├── ci.yml              # Continuous integration
└── train.yml           # Training & promotion

scripts/
└── train_orchestrator.py   # Training orchestration script

registry/
└── production_baseline.json # Current production model metrics

models/
└── {ModelName}/             # Model registry (created by ModelRegistry)
    └── {timestamp}/         # Versioned model files
        ├── model.pkl        # Serialized model
        └── metadata.json    # Model metadata

artifacts/                   # Training artifacts (created during runs)
└── training_report.json     # Detailed training results
```

## Usage

### Running Manual Training

1. Go to **Actions** tab in GitHub
2. Select **Train Models** workflow
3. Click **Run workflow**
4. Choose parameters:
   - Branch: `main` or `develop`
   - Model type: `both`, `xgboost`, or `lightgbm`
   - Trials: Number for hyperparameter optimization
5. Click **Run workflow**

### Monitoring Training

- View workflow runs in the **Actions** tab
- Each run creates a detailed summary with:
  - Model performance metrics
  - Promotion decision and reason
  - Links to artifacts

### Downloading Artifacts

1. Go to completed workflow run
2. Scroll to **Artifacts** section at bottom
3. Download `training-artifacts-{run_id}`
4. Contains:
   - `artifacts/training_report.json` - Full results
   - `models/` - Trained model files
   - `mlruns/` - MLflow experiment data

## Customization

### Change Training Schedule

Edit `.github/workflows/train.yml`:

```yaml
schedule:
  - cron: '0 2 * * 1'  # Every Monday at 2 AM UTC
```

Cron syntax:
- `'0 2 * * *'` - Daily at 2 AM
- `'0 2 * * 1,4'` - Monday and Thursday at 2 AM
- `'0 */6 * * *'` - Every 6 hours

### Adjust Promotion Criteria

Edit the comparison logic in `train.yml` → "Compare and decide promotion" step.

Current: Promotes if `new_rmse < baseline_rmse`

Options:
- Add minimum improvement threshold: `new_rmse < baseline_rmse * 0.95` (5% improvement)
- Require R² improvement too: `new_rmse < baseline_rmse && new_r2 > baseline_r2`
- Add statistical significance tests

### Change Artifact Storage

Current: GitHub Actions artifacts (90-day retention, free)

For production with long-term storage:

1. **AWS S3:**
   ```yaml
   - name: Upload to S3
     run: |
       aws s3 cp models/ s3://your-bucket/models/ --recursive
   ```

2. **Google Cloud Storage:**
   ```yaml
   - name: Upload to GCS
     run: |
       gsutil -m cp -r models/ gs://your-bucket/models/
   ```

3. **MLflow Model Registry:**
   - Set up MLflow tracking server
   - Update `MLFLOW_TRACKING_URI` in workflow
   - Models automatically logged to MLflow

### Add Notifications

Add to `train.yml`:

```yaml
- name: Notify on Slack
  if: steps.promote.outputs.should_promote == 'true'
  uses: slackapi/slack-github-action@v1
  with:
    payload: |
      {
        "text": "🚀 New model promoted! RMSE: ${{ steps.metrics.outputs.new_rmse }}"
      }
  env:
    SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```

## Secrets Management

For production deployments, add secrets:

1. Go to repository **Settings** → **Secrets and variables** → **Actions**
2. Add:
   - `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` (for S3)
   - `GCP_CREDENTIALS` (for GCS)
   - `MLFLOW_TRACKING_URI` (for remote MLflow)
   - `SLACK_WEBHOOK_URL` (for notifications)

## Troubleshooting

### Workflow fails with "No module named 'rental_prediction'"

- Check that `uv sync` completed successfully
- Ensure package is properly installed: `uv run python -c "import rental_prediction"`

### Model promotion not working

- Check `registry/production_baseline.json` exists and is valid JSON
- Verify `jq` is available (pre-installed on ubuntu-latest)
- Check the comparison logic output in workflow logs

### Artifacts not uploading

- Ensure paths exist: `mkdir -p artifacts models mlruns`
- Check artifact size (max 10GB per workflow)
- Verify retention days <= 90

## Best Practices

1. **Test locally first:**
   ```bash
   python scripts/train_orchestrator.py --model-type both --n-trials 10
   ```

2. **Monitor workflow costs:**
   - GitHub Actions: 2,000 free minutes/month for private repos
   - Each training run: ~10-20 minutes
   - Weekly runs: ~80 minutes/month ✅

3. **Version control:**
   - Commit `registry/production_baseline.json` changes
   - Tag releases: `git tag v1.0-prod-model`

4. **Review promotions:**
   - Check workflow summaries regularly
   - Investigate unexpected metric changes
   - Keep MLflow UI running for detailed analysis

## Example Workflow Run

```
🎯 Training Results
Run ID: 1234567890
Timestamp: 2024-01-15 02:00:00 UTC

Model Performance
Model         | XGBoostModel
Test RMSE     | 245.67
Test R²       | 0.8234

Promotion Decision
Status: ✅ PROMOTED
Reason: RMSE improved by 3.45%

The new model has been promoted to production! 🚀
```
