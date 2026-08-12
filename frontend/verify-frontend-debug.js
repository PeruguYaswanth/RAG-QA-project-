const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');
const filePath = path.resolve(__dirname, '..', 'backend', 'data', 'uploads', '072adc83-eca-4423-b0cc-6e0210277f07.pdf');

(async () => {
  console.log('Using file:', filePath);
  if (!fs.existsSync(filePath)) {
    console.error('File not found');
    process.exit(1);
  }

  const browser = await chromium.launch();
  const page = await browser.newPage();

  page.on('console', msg => console.log('PAGE LOG>', msg.text()));
  page.on('pageerror', err => console.log('PAGE ERROR>', err.message));
  page.on('requestfailed', req => console.log('REQUEST FAILED>', req.url(), req.failure()?.errorText));
  page.on('requestfinished', req => {
    if (req.url().includes('/api/upload') || req.url().includes('/api/ask')) {
      console.log('REQUEST FINISHED>', req.method(), req.url(), req.response()?.status());
    }
  });

  await page.goto('http://localhost:5173/', { waitUntil: 'networkidle' });
  const input = page.locator('input[type=file]');
  const count = await input.count();
  console.log('file input count', count);
  await input.setInputFiles(filePath);
  const files = await page.evaluate(() => {
    const input = document.querySelector('input[type=file]');
    return input ? Array.from(input.files).map(f => f.name) : [];
  });
  console.log('input files', files);

  await page.waitForTimeout(3000);
  await page.fill('textarea[placeholder="Ask a question..."]', 'What is the main topic of this document?');
  await page.click('button:has-text("Send")');

  const response = await page.waitForSelector('text=To-Do List Application', { timeout: 20000 }).catch(() => null);
  if (response) {
    console.log('found answer', await response.textContent());
    await browser.close();
    process.exit(0);
  }

  const full = await page.locator('body').innerText();
  console.log('body snippet:', full.slice(0, 2000));
  await browser.close();
  process.exit(1);
})();
