import React, { memo } from 'react';
import { Handle, Position } from 'reactflow';
import { Brain, Layers, GitBranch, Zap, AlertOctagon, Flag } from 'lucide-react';
import { ROLES, ROLE_COLORS } from '../../utils/constants';

const ROLE_ICONS = {
    [ROLES.INTENT]: Brain,
    [ROLES.CONTEXT]: Layers,
    [ROLES.CONDITION]: GitBranch,
    [ROLES.ACTION]: Zap,
    [ROLES.CONSTRAINT]: AlertOctagon,
    [ROLES.OUTCOME]: Flag,
};

const SemanticBlock = ({ data, selected }) => {
    const { role, expression, status } = data;
    const Icon = ROLE_ICONS[role] || Brain;
    const color = ROLE_COLORS[role] || '#888';

    return (
        <div style={{
            minWidth: '180px',
            maxWidth: '250px',
            backgroundColor: 'hsl(var(--bg-panel))',
            border: `1px solid ${selected ? color : 'hsl(var(--border-subtle))'}`,
            borderRadius: '8px',
            boxShadow: selected ? `0 0 0 2px hsla(from ${color} h s l / 0.3)` : '0 2px 5px rgba(0,0,0,0.2)',
            transition: 'all 0.2s ease',
            position: 'relative',
        }}>
            {/* Input Handle */}
            <Handle
                type="target"
                position={Position.Top}
                style={{ background: 'hsl(var(--text-dim))', width: '8px', height: '8px' }}
            />

            {/* Header: Icon + Role + Status */}
            <div style={{
                display: 'flex',
                alignItems: 'center',
                padding: '8px 12px',
                borderBottom: '1px solid hsla(var(--border-subtle), 0.5)',
                gap: '8px',
                backgroundColor: `hsla(from ${color} h s l / 0.05)`,
                borderTopLeftRadius: '8px',
                borderTopRightRadius: '8px',
            }}>
                <div style={{ color: color }}>
                    <Icon size={14} />
                </div>
                <span style={{
                    fontSize: '0.75rem',
                    fontWeight: 600,
                    textTransform: 'uppercase',
                    color: color,
                    flex: 1
                }}>
                    {role}
                </span>
                {status && (
                    <span style={{
                        fontSize: '0.65rem',
                        padding: '2px 6px',
                        borderRadius: '4px',
                        backgroundColor: 'hsl(var(--bg-app))',
                        color: 'hsl(var(--text-dim))',
                        border: '1px solid hsl(var(--border-subtle))'
                    }}>
                        {status}
                    </span>
                )}
            </div>

            {/* Body: Expression */}
            <div style={{ padding: '10px 12px' }}>
                <p style={{
                    margin: 0,
                    fontSize: '0.9rem',
                    color: expression ? 'hsl(var(--text-primary))' : 'hsl(var(--text-dim))',
                    fontStyle: expression ? 'normal' : 'italic',
                    lineHeight: '1.4',
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word'
                }}>
                    {expression || '(Empty expression)'}
                </p>
            </div>

            {/* Output Handle */}
            <Handle
                type="source"
                position={Position.Bottom}
                style={{ background: 'hsl(var(--text-primary))', width: '8px', height: '8px' }}
            />
        </div>
    );
};

export default memo(SemanticBlock);
