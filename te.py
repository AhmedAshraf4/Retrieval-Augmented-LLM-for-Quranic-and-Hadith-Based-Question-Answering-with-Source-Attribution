from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch_persistent_context(
        user_data_dir="selenium1",
        channel="chrome",
        headless=False
    )

    page = browser.new_page()
    page.goto("https://chat.openai.com")

    # Count existing assistant messages
    initial_count = page.locator(
        '[data-message-author-role="assistant"]'
    ).count()

    # Type message
    page.keyboard.type("Why is Playwright faster now?")

    # SEND (ChatGPT-specific)
    page.keyboard.press("Control+Enter")

    # Wait for new assistant message
    page.wait_for_function(
        """count => {
            return document.querySelectorAll(
                '[data-message-author-role="assistant"]'
            ).length > count
        }""",
        arg=initial_count,
        timeout=60000
    )

    response = page.locator(
        '[data-message-author-role="assistant"]'
    ).last.inner_text()

    print("\nCHATGPT RESPONSE:\n")
    print(response)

    input("\nPress Enter to close the browser...")
    browser.close()
