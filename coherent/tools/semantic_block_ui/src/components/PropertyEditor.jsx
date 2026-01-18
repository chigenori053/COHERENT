import React from 'react';
import { STATUSES } from '../utils/constants';

const PropertyEditor = ({ selectedNode, updateNodeData }) => {
    if (!selectedNode) {
        return (
            <div style={{ color: 'hsl(var(--text-dim))', fontSize: '0.9rem', textAlign: 'center', marginTop: '2rem' }}>
                Select a block to edit its properties
            </div>
        );
    }

    const { id, data } = selectedNode;

    const handleChange = (field, value) => {
        updateNodeData(id, { [field]: value });
    };

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>

            {/* Header Info */}
            <div style={{ fontSize: '0.8rem', color: 'hsl(var(--text-dim))', borderBottom: '1px solid hsl(var(--border-subtle))', paddingBottom: '8px' }}>
                <div>ID: <span style={{ fontFamily: 'monospace' }}>{id.slice(0, 8)}...</span></div>
                <div>Role: <span style={{ color: 'hsl(var(--accent-primary))', fontWeight: 600 }}>{data.role}</span></div>
            </div>

            {/* Expression (Multiline) */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                <label style={{ fontSize: '0.85rem', fontWeight: 500, color: 'hsl(var(--text-secondary))' }}>
                    Expression
                </label>
                <textarea
                    value={data.expression || ''}
                    onChange={(e) => handleChange('expression', e.target.value)}
                    placeholder="Describe the intent, action, or context..."
                    style={{
                        minHeight: '80px',
                        backgroundColor: 'hsl(var(--bg-element))',
                        border: '1px solid hsl(var(--border-subtle))',
                        borderRadius: '6px',
                        padding: '8px',
                        color: 'hsl(var(--text-primary))',
                        fontFamily: 'inherit',
                        resize: 'vertical'
                    }}
                />
            </div>

            {/* Status */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                <label style={{ fontSize: '0.85rem', fontWeight: 500, color: 'hsl(var(--text-secondary))' }}>
                    Status
                </label>
                <select
                    value={data.status || STATUSES.DRAFT}
                    onChange={(e) => handleChange('status', e.target.value)}
                    style={{
                        backgroundColor: 'hsl(var(--bg-element))',
                        border: '1px solid hsl(var(--border-subtle))',
                        borderRadius: '6px',
                        padding: '8px',
                        color: 'hsl(var(--text-primary))',
                        cursor: 'pointer'
                    }}
                >
                    {Object.values(STATUSES).map(s => (
                        <option key={s} value={s}>{s}</option>
                    ))}
                </select>
            </div>

            {/* Confidence */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                <label style={{ fontSize: '0.85rem', fontWeight: 500, color: 'hsl(var(--text-secondary))' }}>
                    Confidence <span style={{ color: 'hsl(var(--text-dim))' }}>({(data.confidence || 0).toFixed(1)})</span>
                </label>
                <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={data.confidence || 0}
                    onChange={(e) => handleChange('confidence', parseFloat(e.target.value))}
                    style={{ width: '100%' }}
                />
            </div>

        </div>
    );
};

export default PropertyEditor;
