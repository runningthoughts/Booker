import React, { useState, useRef, useEffect, useCallback, memo } from 'react'

// SpinUpMessage component - moved outside to prevent recreation
const SpinUpMessage = memo(({ isSpinningUp, spinUpCountdown }) => {
  if (!isSpinningUp) return null

  const progress = ((60 - spinUpCountdown) / 60) * 100

  return (
    <div className="spin-up-overlay">
      <div className="spin-up-message">
        <div className="spin-up-icon">⚙️</div>
        <h3 className="spin-up-title">Waking the hamsters…</h3>
        <p className="spin-up-text">
          our free-tier server takes a catnap after 15 minutes. Give it up to 60 seconds to stretch, yawn, and start answering your questions!
        </p>
        <div className="progress-container">
          <div className="progress-bar">
            <div 
              className="progress-fill" 
              style={{ width: `${progress}%` }}
            ></div>
          </div>
          <div className="countdown-text">{spinUpCountdown}s remaining</div>
        </div>
      </div>
    </div>
  )
})

const BookerChat = () => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'assistant',
      content: 'Hello! I\'m Booker, your AI assistant for book questions. Ask me anything about the books in your library.',
      sources: []
    }
  ])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [bookId, setBookId] = useState('')
  const [bookMetadata, setBookMetadata] = useState(null)
  const [coverImage, setCoverImage] = useState(null)
  const [imageLayout, setImageLayout] = useState('vertical') // 'vertical' or 'horizontal'
  const [isSpinningUp, setIsSpinningUp] = useState(false)
  const [spinUpCountdown, setSpinUpCountdown] = useState(60)
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null) // Add ref for the input field

  // Function to load book assets
  const loadBookAssets = async (bookId) => {
    try {
      // Determine API base URL based on environment
      const apiBaseUrl = window.location.hostname === 'booker-ui.onrender.com' 
        ? 'https://booker-api-56am.onrender.com' 
        : ''
      
      // Get DATA_BASE_URL from backend settings
      const configResponse = await fetch(`${apiBaseUrl}/config`)
      let baseUrl = '/library/' // default for development
      
      if (configResponse.ok) {
        const config = await configResponse.json()
        if (config.is_production && config.data_base_url) {
          baseUrl = config.data_base_url
          if (!baseUrl.endsWith('/')) {
            baseUrl += '/'
          }
        }
      }
      
      const bookUrl = `${baseUrl}${bookId}/assets/`
      
      // Try to load title.json
      try {
        const titleResponse = await fetch(`${bookUrl}title.json`)
        if (titleResponse.ok) {
          const metadata = await titleResponse.json()
          setBookMetadata(metadata)
          setIsSpinningUp(false)
        }
      } catch (error) {
        console.log('No title.json found for book:', bookId)
      }

      // Try to load cover image
      try {
        const coverResponse = await fetch(`${bookUrl}cover.png`)
        if (coverResponse.ok) {
          setCoverImage(`${bookUrl}cover.png`)
          setIsSpinningUp(false)
          
          // Create an image element to check dimensions
          const img = new Image()
          img.onload = () => {
            // Determine layout based on aspect ratio
            const aspectRatio = img.width / img.height
            setImageLayout(aspectRatio > 1.5 ? 'horizontal' : 'vertical')
          }
          img.src = `${bookUrl}cover.png`
        }
      } catch (error) {
        console.log('No cover image found for book:', bookId)
      }
    } catch (error) {
      console.log('Error loading book assets:', error)
    }
  }

  useEffect(() => {
    // Get bookId from URL query parameters (case-insensitive)
    const urlParams = new URLSearchParams(window.location.search)
    let bookIdParam = urlParams.get('bookId') || urlParams.get('bookid') || urlParams.get('BookId') || urlParams.get('BOOKID')
    
    if (bookIdParam) {
      // Convert to lowercase for consistency
      bookIdParam = bookIdParam.toLowerCase()
      setBookId(bookIdParam)
      
      // Show spin up message only on production
      if (window.location.hostname === 'booker-ui.onrender.com') {
        setIsSpinningUp(true)
        setSpinUpCountdown(60)
      }
      
      loadBookAssets(bookIdParam)
    } else {
      // Show error if no bookId provided
      setMessages(prev => [...prev, {
        id: Date.now(),
        type: 'assistant',
        content: 'Error: No book ID specified. Please add ?bookId=YourBookId to the URL.',
        sources: []
      }])
    }
    
    // Focus the input field when component mounts
    setTimeout(() => {
      if (inputRef.current) {
        inputRef.current.focus()
      }
    }, 100) // Small delay to ensure DOM is ready
  }, [])

  // Countdown timer for spin up
  const updateCountdown = useCallback(() => {
    setSpinUpCountdown(prev => {
      if (prev <= 1) {
        setIsSpinningUp(false)
        return 0
      }
      return prev - 1
    })
  }, [])

  useEffect(() => {
    if (!isSpinningUp) return

    const timer = setInterval(updateCountdown, 1000)
    return () => clearInterval(timer)
  }, [isSpinningUp, updateCountdown])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
    
    // Focus the input field after assistant responds (when loading stops)
    if (!isLoading && inputRef.current) {
      setTimeout(() => {
        inputRef.current.focus()
      }, 100) // Small delay to ensure the response is fully rendered
    }
  }, [messages, isLoading])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!inputValue.trim() || isLoading || !bookId) return

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputValue.trim(),
      sources: []
    }

    setMessages(prev => [...prev, userMessage])
    setInputValue('')
    setIsLoading(true)

    try {
      // Determine API base URL based on environment (same logic as loadBookAssets)
      const apiBaseUrl = window.location.hostname === 'booker-ui.onrender.com' 
        ? 'https://booker-api-56am.onrender.com' 
        : ''
      
      const response = await fetch(`${apiBaseUrl}/ask/${bookId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: userMessage.content,
          k: 5
        }),
      })

      console.log('API URL:', `${apiBaseUrl}/ask/${bookId}`)
      console.log('Response status:', response.status)
      console.log('Response headers:', response.headers)

      if (!response.ok) {
        const errorText = await response.text()
        console.error('API Error Response:', errorText)
        throw new Error(`API returned ${response.status}: ${errorText}`)
      }

      // Check if response has content
      const responseText = await response.text()
      console.log('Raw response:', responseText)
      
      if (!responseText.trim()) {
        throw new Error('Empty response from server')
      }

      let data
      try {
        data = JSON.parse(responseText)
      } catch (jsonError) {
        console.error('JSON Parse Error:', jsonError)
        console.error('Response text that failed to parse:', responseText)
        throw new Error(`Invalid JSON response: ${jsonError.message}`)
      }
      
      const assistantMessage = {
        id: Date.now() + 1,
        type: 'assistant',
        content: data.answer,
        sources: data.sources || []
      }

      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
      console.error('Error:', error)
      const errorMessage = {
        id: Date.now() + 1,
        type: 'assistant',
        content: `Error: ${error.message}. Please check the console for more details.`,
        sources: []
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const openPDF = (fileName, pageStart) => {
    // In a real implementation, this would open the PDF at the specific page
    alert(`Opening ${fileName} at page ${pageStart}`)
  }

  const Citation = ({ source }) => (
    <div 
      className="citation"
      onClick={() => openPDF(source.file_name, source.page_start)}
      title={`Click to open ${source.file_name} at page ${source.page_start}`}
    >
      <div className="citation-header">
        <span className="citation-number">[{source.source_id}]</span>
        <span className="citation-file">{source.file_name}</span>
        <span className="citation-pages">Pages {source.page_start}-{source.page_end}</span>
      </div>
      <div className="citation-text">{source.text}</div>
      {source.summary && (
        <div className="citation-summary">
          <strong>Summary:</strong> {source.summary}
        </div>
      )}
    </div>
  )

  const CollapsibleSources = ({ sources }) => {
    const [isExpanded, setIsExpanded] = useState(false)

    if (!sources || sources.length === 0) return null

    return (
      <div className="sources-container">
        <button 
          className="sources-toggle"
          onClick={() => setIsExpanded(!isExpanded)}
        >
          <span className="sources-label">Sources ({sources.length})</span>
          <span className={`sources-arrow ${isExpanded ? 'expanded' : ''}`}>▼</span>
        </button>
        {isExpanded && (
          <div className="sources-content">
            {sources.map((source, index) => (
              <Citation key={index} source={source} />
            ))}
          </div>
        )}
      </div>
    )
  }

  const Message = ({ message }) => (
    <div className={`message ${message.type}`}>
      <div className="message-content">
        {message.content}
      </div>
      <CollapsibleSources sources={message.sources} />
    </div>
  )

  // BookHeader component for displaying cover image and metadata
  const BookHeader = () => {
    if (!bookMetadata && !coverImage) return null

    return (
      <div className={`book-header ${imageLayout}`}>
        {coverImage && (
          <div className="cover-image-container">
            <img src={coverImage} alt="Book Cover" className="cover-image" />
          </div>
        )}
        {bookMetadata && (
          <div className="book-info">
            <div className="book-info-content">
              <h2 className="book-title">{bookMetadata.book_name}</h2>
              {bookMetadata.book_author && (
                <p className="book-author">by {bookMetadata.book_author}</p>
              )}
              {bookMetadata.book_description && (
                <p className="book-description">{bookMetadata.book_description}</p>
              )}
              <div className="book-details">
                {bookMetadata.publisher && (
                  <div className="detail-item">
                    <strong>Publisher:</strong> {bookMetadata.publisher}
                  </div>
                )}
                {bookMetadata.year && (
                  <div className="detail-item">
                    <strong>Year:</strong> {bookMetadata.year}
                  </div>
                )}
                {bookMetadata.isbn && (
                  <div className="detail-item">
                    <strong>ISBN:</strong> {bookMetadata.isbn}
                  </div>
                )}
                {bookMetadata.copyright && (
                  <div className="detail-item">
                    <strong>Copyright:</strong> {bookMetadata.copyright}
                  </div>
                )}
                {bookMetadata.permission && (
                  <div className="detail-item permission-text">
                    <strong>Permission:</strong> {bookMetadata.permission}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="booker-chat">
      <style jsx>{`
        .booker-chat {
          display: flex;
          flex-direction: column;
          height: 100vh;
          max-width: 1200px;
          margin: 0 auto;
          background: white;
          box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
          position: relative;
        }

        .header {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          padding: 1rem 2rem;
          text-align: center;
          box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }

        .header h1 {
          margin: 0;
          font-size: 1.8rem;
          font-weight: 600;
        }

        .header p {
          margin: 0.5rem 0 0 0;
          opacity: 0.9;
          font-size: 0.9rem;
        }

        .messages-container {
          flex: 1;
          overflow-y: auto;
          padding: 1rem;
          background: #f8f9fa;
        }

        .message {
          margin-bottom: 1.5rem;
          max-width: 80%;
        }

        .message.user {
          margin-left: auto;
        }

        .message.assistant {
          margin-right: auto;
        }

        .message-content {
          padding: 1rem 1.5rem;
          border-radius: 18px;
          line-height: 1.5;
          word-wrap: break-word;
        }

        .message.user .message-content {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          margin-left: auto;
        }

        .message.assistant .message-content {
          background: white;
          color: #333;
          border: 1px solid #e1e5e9;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }

        .citation {
          background: #f8f9fa;
          border: 1px solid #e9ecef;
          border-radius: 8px;
          padding: 0.75rem;
          margin-bottom: 0.5rem;
          cursor: pointer;
          transition: all 0.2s ease;
        }

        .citation:hover {
          background: #e9ecef;
          border-color: #667eea;
          transform: translateY(-1px);
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }

        .citation-header {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          margin-bottom: 0.5rem;
          font-size: 0.8rem;
        }

        .citation-number {
          background: #667eea;
          color: white;
          padding: 0.2rem 0.4rem;
          border-radius: 4px;
          font-weight: 600;
        }

        .citation-file {
          font-weight: 600;
          color: #333;
        }

        .citation-pages {
          color: #666;
          margin-left: auto;
        }

        .citation-text {
          font-size: 0.85rem;
          color: #555;
          line-height: 1.4;
          margin-bottom: 0.5rem;
        }

        .citation-summary {
          font-size: 0.8rem;
          color: #666;
          font-style: italic;
        }

        .sources-container {
          margin-top: 1rem;
          padding: 0 1.5rem;
        }

        .sources-toggle {
          background: #f8f9fa;
          border: 1px solid #e9ecef;
          border-radius: 8px;
          padding: 0.75rem 1rem;
          width: 100%;
          display: flex;
          align-items: center;
          justify-content: space-between;
          cursor: pointer;
          transition: all 0.2s ease;
          font-size: 0.9rem;
          font-weight: 600;
          color: #666;
        }

        .sources-toggle:hover {
          background: #e9ecef;
          border-color: #667eea;
        }

        .sources-label {
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .sources-arrow {
          transition: transform 0.2s ease;
          font-size: 0.8rem;
        }

        .sources-arrow.expanded {
          transform: rotate(180deg);
        }

        .sources-content {
          margin-top: 0.5rem;
        }

        .input-container {
          padding: 1rem 2rem 2rem 2rem;
          background: white;
          border-top: 1px solid #e1e5e9;
        }

        .input-form {
          display: flex;
          gap: 0.75rem;
          align-items: flex-end;
        }

        .input-wrapper {
          flex: 1;
          position: relative;
        }

        .input-field {
          width: 100%;
          padding: 1rem 1.25rem;
          border: 2px solid #e1e5e9;
          border-radius: 25px;
          font-size: 1rem;
          outline: none;
          transition: border-color 0.2s ease;
          resize: none;
          min-height: 50px;
          max-height: 120px;
          font-family: inherit;
        }

        .input-field:focus {
          border-color: #667eea;
        }

        .input-field:disabled {
          background: #f8f9fa;
          cursor: not-allowed;
        }

        .send-button {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          border: none;
          border-radius: 50%;
          width: 50px;
          height: 50px;
          cursor: pointer;
          transition: all 0.2s ease;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 1.2rem;
        }

        .send-button:hover:not(:disabled) {
          transform: translateY(-2px);
          box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }

        .send-button:disabled {
          opacity: 0.6;
          cursor: not-allowed;
          transform: none;
        }

        .loading-indicator {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          color: #666;
          font-style: italic;
          margin-bottom: 1rem;
        }

        .loading-dots {
          display: flex;
          gap: 0.2rem;
        }

        .loading-dot {
          width: 6px;
          height: 6px;
          background: #667eea;
          border-radius: 50%;
          animation: bounce 1.4s ease-in-out infinite both;
        }

        .loading-dot:nth-child(1) { animation-delay: -0.32s; }
        .loading-dot:nth-child(2) { animation-delay: -0.16s; }

        @keyframes bounce {
          0%, 80%, 100% {
            transform: scale(0);
          } 40% {
            transform: scale(1);
          }
        }

        @media (max-width: 768px) {
          .booker-chat {
            height: 100vh;
          }
          
          .header {
            padding: 1rem;
          }
          
          .header h1 {
            font-size: 1.5rem;
          }

          .book-header {
            padding: 1rem;
          }

          .book-header.vertical {
            flex-direction: column;
            gap: 1rem;
            text-align: center;
          }

          .book-title {
            font-size: 1.5rem;
          }

          .book-author {
            font-size: 1rem;
          }

          .book-description {
            font-size: 0.9rem;
          }
          
          .message {
            max-width: 95%;
          }
          
          .input-container {
            padding: 1rem;
          }
        }

        .book-header {
          background: white;
          border-bottom: 1px solid #e1e5e9;
          padding: 1.5rem 2rem;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }

        .book-header.vertical {
          display: flex;
          gap: 2rem;
          align-items: flex-start;
        }

        .book-header.horizontal {
          display: flex;
          flex-direction: column;
          align-items: center;
          text-align: center;
        }

        .cover-image-container {
          flex-shrink: 0;
        }

        .cover-image {
          max-height: 500px;
          max-width: 100%;
          border-radius: 8px;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
          object-fit: contain;
        }

        .book-header.horizontal .cover-image {
          max-width: 600px;
          margin-bottom: 1.5rem;
        }

        .book-info {
          flex: 1;
          min-width: 0;
        }

        .book-info-content {
          max-width: 100%;
        }

        .book-title {
          margin: 0 0 0.5rem 0;
          font-size: 1.8rem;
          font-weight: 700;
          color: #2c3e50;
          line-height: 1.2;
        }

        .book-author {
          margin: 0 0 1rem 0;
          font-size: 1.1rem;
          color: #667eea;
          font-weight: 500;
        }

        .book-description {
          margin: 0 0 1.5rem 0;
          font-size: 0.95rem;
          color: #555;
          line-height: 1.6;
        }

        .book-details {
          display: grid;
          gap: 0.5rem;
        }

        .detail-item {
          font-size: 0.9rem;
          color: #666;
          line-height: 1.4;
        }

        .detail-item strong {
          color: #333;
          font-weight: 600;
        }

        .permission-text {
          font-size: 0.8rem;
          color: #777;
          font-style: italic;
          margin-top: 0.5rem;
          padding-top: 0.5rem;
          border-top: 1px solid #eee;
        }

        .spin-up-overlay {
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(255, 255, 255, 0.95);
          backdrop-filter: blur(5px);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 1000;
          animation: fadeIn 0.3s ease-in;
        }

        .spin-up-message {
          background: white;
          border-radius: 16px;
          padding: 2rem;
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
          max-width: 500px;
          text-align: center;
          border: 1px solid #e1e5e9;
        }

        .spin-up-icon {
          font-size: 3rem;
          margin-bottom: 1rem;
          animation: spin 2s linear infinite;
        }

        .spin-up-title {
          margin: 0 0 1rem 0;
          font-size: 1.5rem;
          font-weight: 600;
          color: #333;
        }

        .spin-up-text {
          margin: 0 0 1.5rem 0;
          color: #666;
          line-height: 1.5;
          font-size: 0.95rem;
        }

        .progress-container {
          margin-top: 1.5rem;
        }

        .progress-bar {
          width: 100%;
          height: 8px;
          background: #e1e5e9;
          border-radius: 4px;
          overflow: hidden;
          margin-bottom: 0.75rem;
        }

        .progress-fill {
          height: 100%;
          background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
          border-radius: 4px;
          transition: width 1s ease;
        }

        .countdown-text {
          font-size: 0.9rem;
          color: #667eea;
          font-weight: 600;
        }

        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }

        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>

      <div className="header">
        <h1>📚 Booker</h1>
        <p>Ask questions about your books and get intelligent answers with citations</p>
      </div>

      <BookHeader />

      <div className="messages-container">
        {messages.map((message) => (
          <Message key={message.id} message={message} />
        ))}
        
        {isLoading && (
          <div className="loading-indicator">
            <span>Thinking</span>
            <div className="loading-dots">
              <div className="loading-dot"></div>
              <div className="loading-dot"></div>
              <div className="loading-dot"></div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      <div className="input-container">
        <form onSubmit={handleSubmit} className="input-form">
          <div className="input-wrapper">
            <textarea
              ref={inputRef}
              className="input-field"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Ask a question about your books..."
              disabled={isLoading}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  handleSubmit(e)
                }
              }}
            />
          </div>
          <button
            type="submit"
            className="send-button"
            disabled={isLoading || !inputValue.trim()}
          >
            ➤
          </button>
        </form>
      </div>

      <SpinUpMessage isSpinningUp={isSpinningUp} spinUpCountdown={spinUpCountdown} />
    </div>
  )
}

export default BookerChat 