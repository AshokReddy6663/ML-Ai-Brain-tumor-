import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={'width': 1280, 'height': 900})
        
        await page.goto('http://127.0.0.1:8000')
        await page.wait_for_selector('#performance-section', state='visible', timeout=5000)
        
        # Scroll to the performance section
        await page.evaluate("document.getElementById('performance-section').scrollIntoView()")
        await asyncio.sleep(1) # wait for scrolling/rendering
        
        element = await page.query_selector('#performance-section')
        await element.screenshot(path='ui_performance_section.png')
        
        await browser.close()

if __name__ == '__main__':
    asyncio.run(main())
