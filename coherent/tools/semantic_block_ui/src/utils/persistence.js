import { v4 as uuidv4 } from 'uuid';
import { ROLES, STATUSES, EDGE_TYPES } from './constants';

const SCHEMA_VERSION = '0.1.0';

export const exportToJSON = (nodes, edges) => {
    const blocks = nodes.map((node) => ({
        id: node.id,
        role: node.data.role,
        expression: node.data.expression || '',
        status: node.data.status || STATUSES.DRAFT,
        confidence: node.data.confidence,
        position: {
            x: Math.round(node.position.x),
            y: Math.round(node.position.y),
        },
        created_at: node.data.created_at || new Date().toISOString(),
        // Optional fields
        domain: node.data.domain,
        tags: node.data.tags,
        notes: node.data.notes,
    }));

    const docEdges = edges.map((edge) => ({
        id: edge.id,
        type: edge.data?.type || EDGE_TYPES.SEQUENCE, // Default to sequence if not set
        from: edge.source,
        to: edge.target,
        created_at: edge.data?.created_at || new Date().toISOString(),
    }));

    return {
        schema_version: SCHEMA_VERSION,
        doc_id: uuidv4(),
        title: 'Semantic Block Draft',
        created_at: new Date().toISOString(),
        blocks,
        edges: docEdges,
    };
};

export const importFromJSON = (json) => {
    if (!json || !json.blocks) {
        throw new Error('Invalid JSON: missing blocks');
    }

    const nodes = json.blocks.map((block) => ({
        id: block.id,
        type: 'semanticBlock',
        position: block.position || { x: 0, y: 0 },
        data: {
            role: block.role,
            expression: block.expression,
            status: block.status,
            confidence: block.confidence,
            created_at: block.created_at,
            domain: block.domain,
            tags: block.tags,
            notes: block.notes,
        },
    }));

    const edges = (json.edges || []).map((edge) => ({
        id: edge.id,
        source: edge.from,
        target: edge.to,
        type: 'default', // ReactFlow edge type
        data: {
            type: edge.type,
            created_at: edge.created_at,
        },
        markerEnd: { type: 'arrowclosed' }, // Standardize marker
    }));

    return { nodes, edges };
};
