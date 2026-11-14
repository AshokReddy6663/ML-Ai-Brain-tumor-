import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={'width': 1280, 'height': 800})
        
        await page.goto('http://127.0.0.1:8000')
        await page.screenshot(path='ui_landing.png')
        
        # Click the sample image button
        await page.click('button:has-text("Glioma Scan (PNG)")')
        await asyncio.sleep(0.5)
        
        # Click Analyze
        await page.click('button:has-text("Analyze MRI")')
        
        # Wait for the dashboard to appear
        await page.wait_for_selector('#dashboard-container', state='visible', timeout=15000)
        await asyncio.sleep(1) # let animations finish
        
        await page.screenshot(path='ui_dashboard.png', full_page=True)
        await browser.close()

if __name__ == '__main__':
    asyncio.run(main())
