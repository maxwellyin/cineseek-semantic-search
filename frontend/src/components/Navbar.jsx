function Navbar({ currentView, onSearch, onAbout }) {
  return (
    <nav className="brand-nav">
      <a className="brand-mark" href="/search" onClick={(event) => { event.preventDefault(); onSearch(); }}>
        <span className="brand-dot"></span>
        <span>
          <span className="brand-title">CineSeek</span>
          <p className="brand-subtitle">LLM-powered semantic movie search</p>
        </span>
      </a>

      <div className="nav-links">
        <a
          className={`nav-pill ${currentView === 'search' || currentView === 'detail' || currentView === 'results' ? 'active' : ''}`}
          href="/search"
          onClick={(event) => {
            event.preventDefault();
            onSearch();
          }}
        >
          Search
        </a>
        <a
          className={`nav-pill ${currentView === 'about' ? 'active' : ''}`}
          href="/about"
          onClick={(event) => {
            event.preventDefault();
            onAbout();
          }}
        >
          Project Overview
        </a>
        <a
          className="nav-pill"
          href="https://github.com/maxwellyin/cineseek-semantic-search"
          target="_blank"
          rel="noreferrer"
        >
          GitHub
        </a>
      </div>

      <style
        dangerouslySetInnerHTML={{
          __html: `
            .brand-nav {
              display: flex;
              flex-direction: row;
              align-items: center;
              justify-content: space-between;
              backdrop-filter: blur(12px);
              background: rgba(251, 248, 242, 0.76);
              border: 1px solid var(--line);
              border-radius: 24px;
              box-shadow: var(--shadow);
              padding: 14px 18px;
              margin-bottom: 24px;
              gap: 12px;
            }

            .brand-mark {
              display: inline-flex;
              align-items: center;
              gap: 12px;
              text-decoration: none;
              color: var(--ink);
              padding: 0;
              text-align: left;
            }

            .brand-dot {
              width: 14px;
              height: 14px;
              border-radius: 999px;
              background: linear-gradient(135deg, var(--accent), #f1b24c);
              box-shadow: 0 0 0 6px rgba(198, 93, 46, 0.12);
            }

            .brand-title {
              font-size: 1.35rem;
              font-weight: 700;
              letter-spacing: 0.01em;
              display: block;
            }

            .brand-subtitle {
              color: var(--muted);
              font-size: 0.92rem;
              margin: 0;
              font-family: Inter, ui-sans-serif, system-ui, sans-serif;
            }

            .nav-links {
              display: flex;
              flex-wrap: wrap;
              gap: 8px;
            }

            .nav-pill {
              color: var(--muted);
              text-decoration: none;
              border-radius: 999px;
              padding: 8px 14px;
              transition: all 0.18s ease;
              font-family: Inter, ui-sans-serif, system-ui, sans-serif;
              font-size: 0.95rem;
              background: transparent;
              border: none;
              cursor: pointer;
            }

            .nav-pill:hover,
            .nav-pill.active {
              color: var(--ink);
              background: rgba(24, 34, 47, 0.06);
            }

            @media (max-width: 768px) {
              .brand-nav {
                flex-direction: column;
                align-items: flex-start;
              }
            }
          `,
        }}
      />
    </nav>
  );
}

export default Navbar;
