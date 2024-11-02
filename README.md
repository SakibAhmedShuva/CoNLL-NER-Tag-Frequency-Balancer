# CoNLL NER Tag Frequency Balancer

A powerful Flask API and Python tool for balancing Named Entity Recognition (NER) tag frequencies in CoNLL format datasets. This tool helps create more balanced training datasets by adjusting the frequency of NER tags to match desired target frequencies while maintaining sentence integrity.

## Features

- REST API for easy integration
- Detailed frequency statistics and analysis
- Configurable target frequencies
- Maintains sentence integrity
- JSON response with comprehensive statistics
- Support for custom output filenames
- Automatic cleanup of temporary files

## Installation

```bash
git clone https://github.com/SakibAhmedShuva/CoNLL-NER-Tag-Frequency-Balancer.git
cd CoNLL-NER-Tag-Frequency-Balancer
pip install -r requirements.txt
```

## API Usage

### Start the Server

```bash
python app.py
```

### Balance Tags Endpoint

**Endpoint:** `POST /balance`

**Request Format:**
```bash
curl -X POST http://localhost:5000/balance \
  -F "file=@input.conll" \
  -F 'target_frequencies={"B-ORG": 350, "I-ORG": 350, "B-PER": 350, "I-PER": 350, "B-LOC": 350, "I-LOC": 350, "B-MISC": 350, "I-MISC": 350}'
```

**Python Example:**
```python
import requests
import json

url = 'http://localhost:5000/balance'
target_frequencies = {
    'B-ORG': 350, 'I-ORG': 350,
    'B-PER': 350, 'I-PER': 350,
    'B-LOC': 350, 'I-LOC': 350,
    'B-MISC': 350, 'I-MISC': 350
}

files = {'file': open('input.conll', 'rb')}
data = {'target_frequencies': json.dumps(target_frequencies)}

response = requests.post(url, files=files, data=data)
result = response.json()
```

**Response Format:**
```json
{
    "message": "File processed successfully",
    "output_file": "outputs/balanced_input.conll",
    "tag_frequencies": {
        "B-LOC": {
            "count": 479,
            "target": 350,
            "diff": 129
        },
        "B-MISC": {
            "count": 385,
            "target": 350,
            "diff": 35
        }
        // ... other tags
    },
    "formatted_frequencies": [
        "B-LOC: 479 (target: 350, diff: 129)",
        "B-MISC: 385 (target: 350, diff: 35)"
        // ... other formatted strings
    ],
    "summary": {
        "total_sentences": 1250,
        "total_tags": 3324
    }
}
```

### Health Check Endpoint

**Endpoint:** `GET /health`

```bash
curl http://localhost:5000/health
```

**Response:**
```json
{
    "status": "healthy"
}
```

## Configuration

The API can be configured through environment variables or by modifying `app.py`:

- `UPLOAD_FOLDER`: Directory for temporary uploaded files
- `OUTPUT_FOLDER`: Directory for balanced output files
- `MAX_CONTENT_LENGTH`: Maximum allowed file size (default: 16MB)

## Algorithm Details

The balancer uses an iterative refinement approach to achieve target frequencies:

1. Initial sentence selection
2. Iterative removal of sentences contributing to over-represented tags
3. Addition of sentences helping under-represented tags
4. Score-based optimization
5. Best solution tracking across iterations

## License

This project is open source and available under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
