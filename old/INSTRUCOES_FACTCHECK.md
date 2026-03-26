# Instruções para Rodar Simulação com Fact-Checking Reduzido

## Resumo das Correções no Modelo

O código foi corrigido para refletir corretamente o modelo descrito no paper:

1. **Função de Utilidade**: Agora implementa corretamente `γ(R) = γ₀ · σ(R)` onde `σ(R) = 1/(1 + e^(-R))`. A reputação usada é a do próprio agente (não do sender).

2. **Atualização de Reputação**: Corrigida para usar `γ(R_i^t)` em vez de um fator logarítmico que não estava no modelo original.

## Parâmetro de Fact-Checking

O fact-checking é controlado pelo parâmetro `truth_revelation_prob` no arquivo de configuração da simulação. Este parâmetro representa a probabilidade de que a verdade de uma mensagem seja diretamente observável pelos agentes.

- **Valores baixos (0.1-0.3)**: Simulam plataformas tipo **Telegram** com fact-checking reduzido
- **Valores altos (0.7-0.9)**: Simulam plataformas tipo **WhatsApp** com fact-checking mais frequente
- **Valor padrão (1.0)**: Fact-checking sempre ativo (todos os agentes sempre observam a verdade)

## Como Rodar a Simulação com Fact-Checking Reduzido

### Opção 1: Usar arquivo de configuração pré-configurado

Um arquivo de exemplo foi criado em `config/config_sim_low_factcheck.json` com `truth_revelation_prob: 0.1`:

```bash
cd code
python run_simulation.py \
    --config-sim config/config_sim_low_factcheck.json \
    --config-agent config/config_sim_agent.json \
    --config-influencer config/config_sim_influencer.json \
    --config-bot config/config_sim_bot.json \
    --seed 42
```

### Opção 2: Modificar arquivo de configuração existente

Edite o arquivo `config/config_sim_general.json` e adicione o parâmetro `truth_revelation_prob` na seção `message`:

```json
{
    "network_pickle_file": "data/facebook_community_100nodes_seed42.pkl",
    "num_rounds": 250,
    "num_initial_senders": 10,
    "message": {
        "left_bias": 0.5,
        "right_bias": 0.5,
        "prob_truth": 0.6,
        "truth_revelation_prob": 0.1
    }
}
```

Depois execute:

```bash
cd code
python run_simulation.py \
    --config-sim config/config_sim_general.json \
    --config-agent config/config_sim_agent.json \
    --config-influencer config/config_sim_influencer.json \
    --config-bot config/config_sim_bot.json \
    --seed 42
```

### Opção 3: Comparar diferentes níveis de fact-checking

Para comparar os efeitos de diferentes níveis de fact-checking, você pode rodar múltiplas simulações:

```bash
# Fact-checking baixo (Telegram-like)
python run_simulation.py \
    --config-sim config/config_sim_low_factcheck.json \
    --config-agent config/config_sim_agent.json \
    --config-influencer config/config_sim_influencer.json \
    --config-bot config/config_sim_bot.json \
    --seed 42

# Fact-checking alto (WhatsApp-like)
python run_simulation.py \
    --config-sim config/config_sim_high_factcheck.json \
    --config-agent config/config_sim_agent.json \
    --config-influencer config/config_sim_influencer.json \
    --config-bot config/config_sim_bot.json \
    --seed 42
```

## Interpretação dos Resultados

Segundo o paper, quando o fact-checking é reduzido (valores baixos de `truth_revelation_prob`):

1. **Maior contaminação por desinformação**: Os agentes dependem mais de sinais de reputação e proximidade ideológica, levando a maiores taxas de encaminhamento de mensagens falsas.

2. **Mecanismo de reputação fortalecido**: Agentes com alta reputação ganham uma vantagem maior, pois sua reputação serve como sinal crucial para outros agentes.

3. **Câmaras de eco ideológicas**: Com fact-checking limitado, o alinhamento ideológico se torna um sinal primário de crença, criando dinâmicas mais fortes de câmara de eco.

4. **Vulnerabilidade a bots**: Agentes de baixa reputação (bots) podem explorar o sistema mais facilmente, já que outros agentes não podem verificar facilmente a verdade.

## Valores Recomendados para Experimentos

- **Telegram (baixo fact-checking)**: `truth_revelation_prob: 0.1` a `0.2`
- **WhatsApp (alto fact-checking)**: `truth_revelation_prob: 0.7` a `0.9`
- **Sem fact-checking**: `truth_revelation_prob: 0.0` (agentes nunca observam verdade diretamente)
- **Fact-checking completo**: `truth_revelation_prob: 1.0` (padrão, sempre observável)

## Notas Importantes

- O parâmetro `truth_revelation_prob` é aplicado por rodada: a cada rodada, há uma probabilidade `truth_revelation_prob` de que a verdade seja revelada para todos os agentes.
- Quando a verdade não é revelada, os agentes formam crenças usando proximidade ideológica e reputação do sender.
- A atualização de reputação só ocorre quando a verdade é revelada (após a rodada).


