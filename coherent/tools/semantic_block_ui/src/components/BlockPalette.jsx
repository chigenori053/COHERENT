import React from 'react';
import { Brain, Layers, GitBranch, Zap, AlertOctagon, Flag } from 'lucide-react';
import { ROLES, ROLE_COLORS } from '../utils/constants';

const ROLE_ICONS = {
    [ROLES.INTENT]: Brain,
    [ROLES.CONTEXT]: Layers,
    [ROLES.CONDITION]: GitBranch,
    [ROLES.ACTION]: Zap,
    [ROLES.CONSTRAINT]: AlertOctagon,
    [ROLES.OUTCOME]: Flag,
};

const BlockPalette = () => {
    const onDragStart = (event, role) => {
        event.dataTransfer.setData('application/reactflow', role);
        event.dataTransfer.effectAllowed = 'move';
    };

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {Object.values(ROLES).map((role) => {
                const Icon = ROLE_ICONS[role];
                const color = ROLE_COLORS[role];

                return (
                    <div
                        key={role}
                        onDragStart={(event) => onDragStart(event, role)}
                        draggable
                        style={{
                            padding: '10px 12px',
                            borderRadius: '6px',
                            backgroundColor: 'hsl(var(--bg-element))',
                            border: '1px solid transparent',
                            cursor: 'grab',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '10px',
                            transition: 'all 0.2s ease'
                        }}
                        onMouseOver={(e) => {
                            e.currentTarget.style.backgroundColor = 'hsl(var(--bg-element-hover))';
                            e.currentTarget.style.borderColor = color;
                        }}
                        onMouseOut={(e) => {
                            e.currentTarget.style.backgroundColor = 'hsl(var(--bg-element))';
                            e.currentTarget.style.borderColor = 'transparent';
                        }}
                    >
                        <div style={{
                            color: color,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            backgroundColor: `hsla(from ${color} h s l / 0.1)`,
                            padding: '6px',
                            borderRadius: '4px'
                        }}>
                            <Icon size={18} />
                        </div>
                        <span style={{ fontSize: '0.9rem', fontWeight: 500 }}>{role}</span>
                    </div>
                );
            })}
        </div>
    );
};

export default BlockPalette;
