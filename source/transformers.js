
// import * as python from './python.js';
// import * as safetensors from './safetensors.js';

const transformers = {};

transformers.ModelFactory = class {

    async match(context) {
        const obj = await context.peek('json');
        if (obj) {
            if (obj.model_type && obj.architectures) {
                return context.set('transformers.config', obj);
            }
            if (obj.version && obj.added_tokens && obj.model) {
                return context.set('transformers.tokenizer', obj);
            }
            if (obj.tokenizer_class ||
                (obj.bos_token && obj.eos_token && obj.unk_token) ||
                (obj.pad_token && obj.additional_special_tokens) ||
                obj.special_tokens_map_file || obj.full_tokenizer_file) {
                return context.set('transformers.tokenizer.config', obj);
            }
            if (context.identifier === 'vocab.json' && Object.keys(obj).length > 256) {
                return context.set('transformers.vocab', obj);
            }
        }
        return null;
    }

    async open(context) {
        const fetch = async (name) => {
            try {
                const content = await context.fetch(name);
                await this.match(content);
                if (content.value) {
                    return content;
                }
            } catch {
                // continue regardless of error
            }
            return null;
        };
        switch (context.type) {
            case 'transformers.config': {
                const tokenizer = await fetch('tokenizer.json');
                const tokenizer_config = await fetch('tokenizer_config.json');
                const vocab = await fetch('vocab.json');
                return new transformers.Model(context, tokenizer, tokenizer_config, vocab);
            }
            case 'transformers.tokenizer': {
                const config = await fetch('config.json');
                const tokenizer_config = await fetch('tokenizer_config.json');
                const vocab = await fetch('vocab.json');
                return new transformers.Model(config, context, tokenizer_config, vocab);
            }
            case 'transformers.tokenizer.config': {
                const config = await fetch('config.json');
                const tokenizer = await fetch('tokenizer.json');
                const vocab = await fetch('vocab.json');
                return new transformers.Model(config, tokenizer, context, vocab);
            }
            case 'transformers.vocab': {
                const config = await fetch('config.json');
                const tokenizer = await fetch('tokenizer.json');
                const tokenizer_config = await fetch('tokenizer_config.json');
                return new transformers.Model(config, tokenizer, tokenizer_config, context);
            }
            default: {
                throw new transformers.Error(`Unsupported Transformers format '${context.type}'.`);
            }
        }
    }

    filter(context, type) {
        return context.type !== 'transformers.config' || (type !== 'transformers.tokenizer' && type !== 'transformers.tokenizer.config' && type !== 'transformers.vocab' && type !== 'safetensors.json');
    }
};

transformers.Model = class {

    constructor(config, tokenizer, tokenizer_config, vocab) {
        this.modules = [];
        this.metadata = [];
        if (config) {
            this.format = 'Transformers';
            this.modules.push(new transformers.Graph(config));
        }
        if (tokenizer || tokenizer_config) {
            this.format = this.format || 'Transformers Tokenizer';
            this.modules.push(new transformers.Tokenizer(tokenizer, tokenizer_config));
        }
        if (vocab) {
            this.format = this.format || 'Transformers Vocabulary';
            this.modules.push(new transformers.Vocabulary(vocab));
        }
    }
};

transformers.Graph = class {

    constructor(context) {
        this.type = 'graph';
        this.name = context.identifier;
        this.nodes = [];
        this.inputs = [];
        this.outputs = [];
        this.metadata = [];
        for (const [key, value] of Object.entries(context.value)) {
            const argument = new transformers.Argument(key, value);
            this.metadata.push(argument);
        }
    }
};

transformers.Tokenizer = class {

    constructor(tokenizer, tokenizer_config) {
        this.type = 'tokenizer';
        this.name = (tokenizer || tokenizer_config).identifier;
    }
};

transformers.Vocabulary = class {

    constructor(context) {
        this.type = 'vocabulary';
        this.name = context.identifier;
    }
};

transformers.Argument = class {

    constructor(name, value) {
        this.name = name;
        this.value = value;
    }
};

transformers.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Transformers model.';
    }
};

export const ModelFactory = transformers.ModelFactory;
