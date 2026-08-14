import { test, expect } from '@playwright/test';

test.describe('Fraud Detection Dashboard', () => {
  test('Successful browser → React → FastAPI → XGBoost prediction', async ({ page }) => {
    // 1. Navigate to dashboard
    await page.goto('/');

    // 2. Wait for health check connection
    await expect(page.locator('.status-text')).toContainText('System Online', { timeout: 10000 });

    // 3. Fill transaction data (using synthetic test data)
    await page.fill('input[name="Time"]', '406.0');
    await page.fill('input[name="Amount"]', '150.0');
    
    // Fill the first few fields with specific values and others with 0.1 to pass validation
    await page.fill('input[name="V1"]', '-2.312');
    await page.fill('input[name="V2"]', '1.951');
    await page.fill('input[name="V3"]', '-1.609');

    for (let i = 4; i <= 28; i++) {
        await page.fill(`input[name="V${i}"]`, '0.1');
    }

    // 4. Submit
    await page.click('button[type="submit"]');

    // 5. Verify results
    // We don't assert exact probability, but we verify the response elements exist and logic holds
    await expect(page.locator('.result-container')).toBeVisible({ timeout: 10000 });
    
    // We expect the threshold to be exactly 31.00%
    await expect(page.locator('.stat-box:has-text("Operating Threshold") .stat-value')).toContainText('31.00%');
    
    // The active model must be accurate
    await expect(page.locator('.model-name')).toContainText('Phase 4 XGBoost Baseline');

    // Verify it either flagged as fraud or legitimate cleanly
    const isFraud = await page.locator('.result-badge').textContent() === 'FRAUDULENT';
    const isLegit = await page.locator('.result-badge').textContent() === 'LEGITIMATE';
    
    expect(isFraud || isLegit).toBeTruthy();
  });

  test('Controlled backend-unavailable scenario', async ({ page }) => {
    // We simulate an unavailable backend by intercepting the /predict endpoint and forcing a failure
    await page.route('**/api/v1/predict', route => route.abort('failed'));

    await page.goto('/');

    await page.fill('input[name="Time"]', '406.0');
    await page.fill('input[name="Amount"]', '150.0');
    
    for (let i = 1; i <= 28; i++) {
        await page.fill(`input[name="V${i}"]`, '0.1');
    }

    await page.click('button[type="submit"]');

    // It should render an API error banner rather than crashing
    await expect(page.locator('.api-error-banner')).toBeVisible({ timeout: 5000 });
    await expect(page.locator('.api-error-banner')).toContainText('Network error');
  });
});
