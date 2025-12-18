
"""
CLI Tool for Ingesting Knowledge into Optical Memory.
Converts YAML rule files into a persistable Optical Hologram.

Usage:
    python -m coherent.tools.ingest_knowledge --input ./knowledge_rules --output ./optical_memory.pt
"""

import argparse
from pathlib import Path
import logging
import sys

# Ensure project root is in path
sys.path.append(str(Path(__file__).parent.parent.parent))

from coherent.engine.knowledge_registry import KnowledgeRegistry
from coherent.memory.factory import get_vector_store
from coherent.engine.symbolic_engine import SymbolicEngine

def main():
    parser = argparse.ArgumentParser(description="Ingest Knowledge into Optical Store")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input directory containing YAML rules")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output path for the optical store file")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input directory {input_path} does not exist.")
        return

    # Setup Logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("IngestTool")
    
    logger.info(f"Initializing Optical Store...")
    # This initializes the singleton store
    store = get_vector_store()
    
    logger.info(f"Loading rules from {input_path}...")
    # Initialize Engine (required for Registry)
    engine = SymbolicEngine()
    
    # Initialize Registry
    # This triggers the auto-indexing logic we implemented in KnowledgeRegistry
    registry = KnowledgeRegistry(root_path=input_path, engine=engine)
    
    count = store.current_count
    logger.info(f"Ingestion complete. Total items in memory: {count}")
    
    logger.info(f"Saving Optical Hologram to {output_path}...")
    store.save(str(output_path))
    
    logger.info("Done.")

if __name__ == "__main__":
    main()
