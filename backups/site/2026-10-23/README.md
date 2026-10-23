# Site experience update · 2026-10-23

Files here are the original theme config, custom CSS, and entry pages before this update.
The complete prior source is recoverable from commit adefbc4.

## Changes

- Inline navigation SVG with persistent Chinese labels, active page semantics, mobile navigation/search bar.
- Editorial homepage, three topic tracks, DSA lab entry, consistent light/dark reader styles.
- Reading progress, font size, focus mode, copy link, native image dialog, reduced-motion and print styles.
- Lazy native search with title-weighted results, safe text rendering, keyboard access and network retry.
- Local copies of jQuery/lazyload with licenses; disable speculative page loading and scroll-reveal.
- Raw pass-through for static CSS/JS/images/HTML; fixes code escaping and legacy standalone MLA page output.
- Conditional MathJax, with math detection for unflagged legacy Markdown.
- Review index rebuilt from existing articles; broken entry-page links repaired.

## Verification

- Hexo production build succeeds.
- Static outputs match the original CSS/JS/vendor and both standalone HTML files byte-for-byte.
- Chromium: 1440 px desktop and 390 px mobile; no horizontal overflow on tested home/article pages.
- Eight navigation items render inline without external icon requests.
- No search index request before search input; search ranks Lightning Indexer article first.
- Keyboard results, Escape clearing, simulated network failure and retry pass.
- Persistent light/dark theme, large text, focus mode and image dialog pass.
- DSA article: 73 display math blocks, 242 rendered MathJax containers; no browser JavaScript errors in primary flow.
- Local links from homepage and seven main navigation pages resolve to generated files.
- Independent DSA lab remains functional and outside the theme.

## Maintenance

Custom build enhancements live in scripts/site-experience.js. Browser behavior lives in
source/js/site-experience.js. Neither requires editing the Stellar theme checkout.
When changing these assets, bump the version query in _config.stellar.yml.
Third-party comments, visit counts and MathJax remain external services.
