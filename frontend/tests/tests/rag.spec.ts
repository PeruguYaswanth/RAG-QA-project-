import { test, expect } from '@playwright/test';

test('test', async ({ page }) => {
  await page.goto('http://localhost:5173/');
  await page.getByText('Drag & drop your document').click();
  await page.locator('input[type="file"]').setInputFiles('CAP Application Print.pdf');
  await page.getByRole('textbox', { name: 'Ask a question...' }).click();
  await page.getByRole('textbox', { name: 'Ask a question...' }).fill('give me colleges only starts with S from the PDF');
  await page.getByRole('textbox', { name: 'Ask a question...' }).click();
  await page.getByRole('textbox', { name: 'Ask a question...' }).fill('give me branches of Siddharth Institute of Engineering and Technology puttur');
});