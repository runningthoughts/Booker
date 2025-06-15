# Booker Makefile

.PHONY: viz ingest

# Default book ID for commands that need one
BOOK_ID ?= ""

viz:
	booker-viz $(BOOK_ID)

ingest:
	python -m booker.ingest_book --book-id $(BOOK_ID) 