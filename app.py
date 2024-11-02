from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Set

class NERBalancer:
    def __init__(self, target_frequencies: Dict[str, int], max_iterations: int = 50):
        self.target_frequencies = target_frequencies
        self.max_iterations = max_iterations
        
    def read_conll(self, filename: str) -> List[List[Tuple[str, str]]]:
        """Read CoNLL file and return list of sentences with (word, tag) pairs."""
        sentences = []
        current_sentence = []
        
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('-DOCSTART-') or not line:
                    if current_sentence:
                        sentences.append(current_sentence)
                        current_sentence = []
                    continue
                    
                parts = line.split()
                word = parts[0]
                tag = parts[-1]
                current_sentence.append((word, tag))
                
            if current_sentence:
                sentences.append(current_sentence)
                
        return sentences

    def get_sentence_tag_counts(self, sentence: List[Tuple[str, str]]) -> Dict[str, int]:
        """Count occurrences of each tag in a sentence."""
        counts = defaultdict(int)
        for _, tag in sentence:
            if tag != 'O':
                counts[tag] += 1
        return counts

    def get_current_counts(self, sentences: List[List[Tuple[str, str]]], selected: Set[int], 
                          sentence_tag_counts: List[Dict[str, int]]) -> Dict[str, int]:
        """Get current tag counts from selected sentences."""
        counts = defaultdict(int)
        for idx in selected:
            for tag, count in sentence_tag_counts[idx].items():
                counts[tag] += count
        return counts

    def calculate_frequency_score(self, current_counts: Dict[str, int]) -> float:
        """Calculate how far current frequencies are from targets."""
        score = 0
        for tag, target in self.target_frequencies.items():
            current = current_counts.get(tag, 0)
            score -= abs(current - target)
        return score

    def should_remove_sentence(self, idx: int, sentence_counts: Dict[str, int], 
                             current_counts: Dict[str, int]) -> bool:
        """Determine if removing a sentence would improve overall balance."""
        for tag, count in sentence_counts.items():
            if tag in self.target_frequencies:
                current = current_counts[tag]
                target = self.target_frequencies[tag]
                if current < target:
                    # Don't remove if we're already under target
                    return False
                if current - count < target * 0.8:  # Allow some undershoot
                    # Don't remove if it would put us too far under target
                    return False
        return True

    def balance_dataset(self, sentences: List[List[Tuple[str, str]]]) -> List[List[Tuple[str, str]]]:
        """Balance the dataset through iterative refinement."""
        sentence_tag_counts = [self.get_sentence_tag_counts(sent) for sent in sentences]
        best_selected = set()
        best_score = float('-inf')
        
        # Initial selection
        selected = set(range(len(sentences)))
        current_counts = self.get_current_counts(sentences, selected, sentence_tag_counts)
        
        for iteration in range(self.max_iterations):
            improved = False
            
            # Try removing sentences that contribute to over-represented tags
            for idx in list(selected):
                sentence_counts = sentence_tag_counts[idx]
                if self.should_remove_sentence(idx, sentence_counts, current_counts):
                    selected.remove(idx)
                    for tag, count in sentence_counts.items():
                        current_counts[tag] -= count
                    improved = True
            
            # Try adding sentences that help under-represented tags
            available = set(range(len(sentences))) - selected
            for idx in list(available):
                sentence_counts = sentence_tag_counts[idx]
                can_add = True
                for tag, count in sentence_counts.items():
                    if tag in self.target_frequencies:
                        if current_counts[tag] + count > self.target_frequencies[tag] * 1.1:  # Allow 10% overshoot
                            can_add = False
                            break
                
                if can_add:
                    selected.add(idx)
                    for tag, count in sentence_counts.items():
                        current_counts[tag] += count
                    improved = True
            
            # Calculate score for this iteration
            score = self.calculate_frequency_score(current_counts)
            if score > best_score:
                best_score = score
                best_selected = selected.copy()
            
            if not improved:
                break
            
            # Print progress every 10 iterations
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, current score: {score}")
                self.print_current_stats(current_counts)
        
        # Return best result found
        return [sentences[idx] for idx in sorted(best_selected)]

    def write_conll(self, sentences: List[List[Tuple[str, str]]], output_filename: str):
        """Write sentences back to CoNLL format."""
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write('-DOCSTART- -X- O O\n\n')
            for sentence in sentences:
                for word, tag in sentence:
                    f.write(f"{word} -X- _ {tag}\n")
                f.write("\n")

    def print_current_stats(self, current_counts: Dict[str, int]):
        """Print current tag frequencies and differences from targets."""
        print("\nCurrent tag frequencies:")
        for tag, count in sorted(current_counts.items()):
            target = self.target_frequencies.get(tag, 0)
            diff = count - target if target > 0 else count
            print(f"{tag}: {count} (target: {target}, diff: {diff})")

    def process_file(self, input_filename: str, output_filename: str):
        """Process the entire file."""
        sentences = self.read_conll(input_filename)
        balanced_sentences = self.balance_dataset(sentences)
        self.write_conll(balanced_sentences, output_filename)
        
        # Print final statistics
        tag_counts = defaultdict(int)
        for sentence in balanced_sentences:
            for _, tag in sentence:
                if tag != 'O':
                    tag_counts[tag] += 1
                    
        print("\nFinal tag frequencies:")
        for tag, count in sorted(tag_counts.items()):
            target = self.target_frequencies.get(tag, 0)
            diff = count - target if target > 0 else count
            print(f"{tag}: {count} (target: {target}, diff: {diff})")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload and output directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

def get_tag_frequencies(sentences: List[List[Tuple[str, str]]], target_frequencies: Dict[str, int]) -> Dict[str, Dict]:
    """Calculate current tag frequencies and differences from targets."""
    tag_counts = defaultdict(int)
    for sentence in sentences:
        for _, tag in sentence:
            if tag != 'O':
                tag_counts[tag] += 1
    
    # Format the results
    frequencies = {}
    for tag in sorted(tag_counts.keys()):
        count = tag_counts[tag]
        target = target_frequencies.get(tag, 0)
        diff = count - target if target > 0 else count
        frequencies[tag] = {
            'count': count,
            'target': target,
            'diff': diff
        }
    
    return frequencies

@app.route('/balance', methods=['POST'])
def balance_tags():
    try:
        # Check if file is present in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get target frequencies from request
        target_frequencies = request.form.get('target_frequencies')
        if not target_frequencies:
            return jsonify({'error': 'No target frequencies provided'}), 400
        
        try:
            target_frequencies = json.loads(target_frequencies)
        except json.JSONDecodeError:
            return jsonify({'error': 'Invalid target frequencies format'}), 400
        
        # Get output filename if provided
        output_filename = request.form.get('output_filename')
        if not output_filename:
            output_filename = f'balanced_{secure_filename(file.filename)}'
        else:
            output_filename = secure_filename(output_filename)
        
        # Save uploaded file
        input_filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        file.save(input_path)
        
        # Process file
        balancer = NERBalancer(target_frequencies)
        sentences = balancer.read_conll(input_path)
        balanced_sentences = balancer.balance_dataset(sentences)
        balancer.write_conll(balanced_sentences, output_path)
        
        # Get frequency statistics
        frequencies = get_tag_frequencies(balanced_sentences, target_frequencies)
        
        # Format the frequencies for display
        formatted_frequencies = []
        for tag, stats in frequencies.items():
            formatted_frequencies.append(
                f"{tag}: {stats['count']} (target: {stats['target']}, diff: {stats['diff']})"
            )
        
        # Cleanup input file
        os.remove(input_path)
        
        return jsonify({
            'message': 'File processed successfully',
            'output_file': output_path,
            'tag_frequencies': frequencies,
            'formatted_frequencies': formatted_frequencies,
            'summary': {
                'total_sentences': len(balanced_sentences),
                'total_tags': sum(stats['count'] for stats in frequencies.values())
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)