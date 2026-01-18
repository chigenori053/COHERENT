export const ROLES = {
    INTENT: 'Intent',
    CONTEXT: 'Context',
    CONDITION: 'Condition',
    ACTION: 'Action',
    CONSTRAINT: 'Constraint',
    OUTCOME: 'Outcome',
};

export const ROLE_COLORS = {
    [ROLES.INTENT]: 'hsl(250, 80%, 65%)', // Purple
    [ROLES.CONTEXT]: 'hsl(190, 80%, 60%)', // Cyan
    [ROLES.CONDITION]: 'hsl(35, 90%, 60%)', // Orange/Yellow
    [ROLES.ACTION]: 'hsl(150, 60%, 50%)', // Green
    [ROLES.CONSTRAINT]: 'hsl(0, 70%, 60%)', // Red
    [ROLES.OUTCOME]: 'hsl(300, 70%, 60%)', // Pink
};

export const STATUSES = {
    DRAFT: 'draft',
    PROVISIONAL: 'provisional',
    STABLE: 'stable',
    DEPRECATED: 'deprecated',
};

export const EDGE_TYPES = {
    SEQUENCE: 'sequence',
    DEPENDS_ON: 'depends_on',
};
