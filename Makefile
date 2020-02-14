
test:
	pytest tests -v -m "not slow" --show-capture=log --log-level=INFO