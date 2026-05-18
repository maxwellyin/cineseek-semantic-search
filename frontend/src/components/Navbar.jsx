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
          className="nav-pill nav-github"
          href="https://github.com/maxwellyin/cineseek-semantic-search"
          target="_blank"
          rel="noreferrer"
          aria-label="GitHub Repository"
        >
          <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
            <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/>
          </svg>
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
              backdrop-filter: blur(16px);
              background: rgba(250, 248, 245, 0.85);
              border: 1px solid var(--line);
              border-radius: 24px;
              box-shadow: var(--shadow);
              padding: 12px 20px;
              margin-bottom: 32px;
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
              transition: opacity var(--transition-smooth);
            }

            .brand-mark:hover {
              opacity: 0.9;
            }

            .brand-dot {
              width: 12px;
              height: 12px;
              border-radius: 999px;
              background: linear-gradient(135deg, var(--accent), #f1b24c);
              box-shadow: 0 0 0 6px rgba(198, 93, 46, 0.12);
            }

            .brand-title {
              font-size: 1.3rem;
              font-weight: 800;
              letter-spacing: -0.01em;
              display: block;
              line-height: 1.1;
            }

            .brand-subtitle {
              color: var(--muted);
              font-size: 0.85rem;
              margin: 4px 0 0 0;
              font-family: Inter, ui-sans-serif, system-ui, sans-serif;
              font-weight: 500;
              letter-spacing: -0.01em;
            }

            .nav-links {
              display: flex;
              align-items: center;
              gap: 10px;
            }

            .nav-pill {
              color: var(--muted);
              text-decoration: none;
              border-radius: 999px;
              padding: 10px 18px;
              transition: all var(--transition-smooth);
              font-family: Inter, ui-sans-serif, system-ui, sans-serif;
              font-size: 0.92rem;
              font-weight: 600;
              background: transparent;
              border: none;
              cursor: pointer;
              display: inline-flex;
              align-items: center;
              justify-content: center;
            }

            .nav-pill:hover {
              color: var(--ink);
              background: rgba(24, 34, 47, 0.05);
            }

            .nav-pill.active {
              color: #ffffff;
              background: var(--ink);
              box-shadow: 0 4px 12px rgba(24, 34, 47, 0.15);
            }

            .nav-github {
              padding: 10px;
              color: var(--muted);
              background: rgba(24, 34, 47, 0.03);
            }

            .nav-github:hover {
              color: var(--ink);
              background: rgba(24, 34, 47, 0.08);
              transform: scale(1.05);
            }

            @media (max-width: 768px) {
              .brand-nav {
                flex-direction: column;
                align-items: center;
                gap: 16px;
                padding: 16px;
              }
              .brand-mark {
                flex-direction: column;
                text-align: center;
                gap: 8px;
              }
              .brand-dot {
                box-shadow: 0 0 0 4px rgba(198, 93, 46, 0.12);
              }
            }
          `,
        }}
      />
    </nav>
  );
}

export default Navbar;
