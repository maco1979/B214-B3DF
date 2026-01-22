
import { test } from '@playwright/test';
import { expect } from '@playwright/test';

test('CameraFrameTest_2025-12-26', async ({ page, context }) => {
  
    // Navigate to URL
    await page.goto('http://localhost:3000/ai-control');

    // Click element
    await page.click("button:has-text('摄像头')");
});