import React, { useState, useRef, useEffect } from 'react'

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
  const messagesEndRef = useRef(null)

  // Function to load book assets
  const loadBookAssets = async (bookId) => {
    try {
      // Try to load title.json
      const titleResponse = await fetch(`http://localhost:8000/library/${bookId}/assets/title.json`)
      if (titleResponse.ok) {
        const metadata = await titleResponse.json()
        setBookMetadata(metadata)
      }
    } catch (error) {
      console.log('No title.json found for book:', bookId)
    }

    try {
      // Try to load cover.png
      const coverResponse = await fetch(`http://localhost:8000/library/${bookId}/assets/cover.png`)
      if (coverResponse.ok) {
        const coverUrl = `http://localhost:8000/library/${bookId}/assets/cover.png`
        setCoverImage(coverUrl)
        
        // Create an image element to check dimensions
        const img = new Image()
        img.onload = () => {
          // Determine layout based on aspect ratio
          const aspectRatio = img.width / img.height
          setImageLayout(aspectRatio > 1.5 ? 'horizontal' : 'vertical')
        }
        img.src = coverUrl
      }
    } catch (error) {
      console.log('No cover.png found for book:', bookId)
    }
  }

  useEffect(() => {
    // Get bookId from URL query parameters
    const urlParams = new URLSearchParams(window.location.search)
    const bookIdParam = urlParams.get('bookId')
    if (bookIdParam) {
      setBookId(bookIdParam)
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
  }, [])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

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
      const response = await fetch(`/api/ask/${bookId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: userMessage.content,
          k: 5
        }),
      })

      if (!response.ok) {
        throw new Error('Failed to get response')
      }

      const data = await response.json()
      
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
        content: 'Sorry, I encountered an error while processing your question. Please try again.',
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
    </div>
  )
}

export default BookerChat 