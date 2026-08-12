const { chromium } = require('playwright');
const path = require('path');
const filePath = path.resolve(__dirname, '..', 'backend', 'data', 'uploads', '072adc83-e2ca-4423-b0cc-6e0210277f07.pdf');

(async () => {
  console.log('Using file:', filePath);
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto('http://127.0.0.1:5173/', { waitUntil: 'networkidle' });
  const fileInput = page.locator('input[type=file]');
  await fileInput.setInputFiles(filePath);
  await page.waitForSelector('text=pages', { timeout: 20000 });
  await page.fill('textarea[placeholder="Ask a question..."]', 'What is the main topic of this document?');
  await page.click('button:has-text("Send")');
  const response = await page.waitForSelector('text=To-Do List Application', { timeout: 20000 }).catch(() => null);
  if (response) {
    console.log('frontend verification succeeded: answer found');
    console.log(await response.textContent());
    await browser.close();
    process.exit(0);
  }
  const bodyText = await page.locator('body').innerText();
  console.error('frontend verification failed; page content snippet:', bodyText.slice(0, 1000));
  await browser.close();
  process.exit(1);
})();
