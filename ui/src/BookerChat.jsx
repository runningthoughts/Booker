import React, { useState, useRef, useEffect, useCallback, memo } from 'react'
import './BookerChat.css'

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
        : '/api'
      
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
        : '/api'
      
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