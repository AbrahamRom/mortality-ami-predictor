// Enhanced search functionality
document.addEventListener('DOMContentLoaded', function() {
  // Add keyboard shortcuts
  document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + K to focus search
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
      e.preventDefault();
      const searchInput = document.querySelector('.md-search__input');
      if (searchInput) {
        searchInput.focus();
      }
    }
  });

  // Smooth scroll for anchor links
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      e.preventDefault();
      const target = document.querySelector(this.getAttribute('href'));
      if (target) {
        target.scrollIntoView({
          behavior: 'smooth',
          block: 'start'
        });
      }
    });
  });

  // Add copy feedback for code blocks
  const clipboard = new ClipboardJS('.md-clipboard');
  
  clipboard.on('success', function(e) {
    const button = e.trigger;
    const originalTitle = button.getAttribute('data-clipboard-text');
    
    // Change button text temporarily
    button.setAttribute('data-clipboard-text', '✓ Copied!');
    
    setTimeout(() => {
      button.setAttribute('data-clipboard-text', originalTitle);
    }, 2000);
    
    e.clearSelection();
  });

  // Table of contents highlighting
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      const id = entry.target.getAttribute('id');
      const tocLink = document.querySelector(`.md-nav__link[href="#${id}"]`);
      
      if (tocLink) {
        if (entry.intersectionRatio > 0) {
          tocLink.classList.add('md-nav__link--active');
        } else {
          tocLink.classList.remove('md-nav__link--active');
        }
      }
    });
  });

  // Track all headings
  document.querySelectorAll('h2[id], h3[id], h4[id]').forEach(heading => {
    observer.observe(heading);
  });

  // External link icons
  document.querySelectorAll('a[href^="http"]').forEach(link => {
    if (!link.hostname.includes(window.location.hostname)) {
      link.setAttribute('target', '_blank');
      link.setAttribute('rel', 'noopener noreferrer');
      link.innerHTML += ' <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="16" height="16" style="display: inline; vertical-align: text-bottom;"><path fill="currentColor" d="M3.75 2A1.75 1.75 0 002 3.75v8.5c0 .966.784 1.75 1.75 1.75h8.5A1.75 1.75 0 0014 12.25v-3.5a.75.75 0 00-1.5 0v3.5a.25.25 0 01-.25.25h-8.5a.25.25 0 01-.25-.25v-8.5a.25.25 0 01.25-.25h3.5a.75.75 0 000-1.5h-3.5zM9.5 1.25a.75.75 0 000 1.5h2.94L6.97 8.22a.75.75 0 101.06 1.06l5.47-5.47v2.94a.75.75 0 001.5 0V1.25a.75.75 0 00-.75-.75h-5.5z"/></svg>';
    }
  });

  // Code language labels
  document.querySelectorAll('pre > code[class*="language-"]').forEach(block => {
    const language = block.className.match(/language-(\w+)/);
    if (language) {
      const label = document.createElement('div');
      label.className = 'code-label';
      label.textContent = language[1].toUpperCase();
      label.style.cssText = 'position: absolute; top: 0.5rem; right: 3rem; background: var(--md-accent-fg-color); color: white; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.75rem; font-weight: 600;';
      block.parentElement.style.position = 'relative';
      block.parentElement.insertBefore(label, block);
    }
  });

  // Progress bar for reading
  let progressBar = document.createElement('div');
  progressBar.id = 'reading-progress';
  progressBar.style.cssText = 'position: fixed; top: 0; left: 0; height: 3px; background: var(--md-accent-fg-color); z-index: 1000; transition: width 0.1s ease;';
  document.body.appendChild(progressBar);

  window.addEventListener('scroll', () => {
    const windowHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
    const scrolled = (window.scrollY / windowHeight) * 100;
    progressBar.style.width = scrolled + '%';
  });

  // Add tooltips for abbreviations
  document.querySelectorAll('abbr[title]').forEach(abbr => {
    abbr.style.cursor = 'help';
    abbr.style.borderBottom = '1px dotted var(--md-accent-fg-color)';
  });

  // API documentation enhancements
  document.querySelectorAll('.doc-signature').forEach(signature => {
    signature.style.cursor = 'pointer';
    signature.addEventListener('click', function() {
      const code = this.querySelector('code');
      if (code) {
        const range = document.createRange();
        range.selectNode(code);
        window.getSelection().removeAllRanges();
        window.getSelection().addRange(range);
      }
    });
  });

  // Search result preview enhancement
  const searchResults = document.querySelector('.md-search-result');
  if (searchResults) {
    const observer = new MutationObserver(() => {
      document.querySelectorAll('.md-search-result__article').forEach(result => {
        result.addEventListener('mouseenter', function() {
          this.style.transform = 'translateX(4px)';
        });
        result.addEventListener('mouseleave', function() {
          this.style.transform = 'translateX(0)';
        });
      });
    });
    
    observer.observe(searchResults, { childList: true, subtree: true });
  }

  // Print optimization
  window.addEventListener('beforeprint', () => {
    document.querySelectorAll('.md-sidebar, .md-header, .md-footer').forEach(el => {
      el.style.display = 'none';
    });
  });

  window.addEventListener('afterprint', () => {
    document.querySelectorAll('.md-sidebar, .md-header, .md-footer').forEach(el => {
      el.style.display = '';
    });
  });
});

// Mathematical formula rendering (if MathJax or KaTeX is available)
if (typeof MathJax !== 'undefined') {
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [['$','$'], ['\\(','\\)']],
      displayMath: [['$$','$$'], ['\\[','\\]']]
    }
  });
}

// Analytics helper functions
function trackOutboundLink(url) {
  if (typeof gtag !== 'undefined') {
    gtag('event', 'click', {
      'event_category': 'outbound',
      'event_label': url,
      'transport_type': 'beacon'
    });
  }
}

// Performance monitoring
if ('performance' in window) {
  window.addEventListener('load', () => {
    setTimeout(() => {
      const perfData = window.performance.timing;
      const pageLoadTime = perfData.loadEventEnd - perfData.navigationStart;
      console.log(`Page load time: ${pageLoadTime}ms`);
    }, 0);
  });
}
