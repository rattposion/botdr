"""
Sistema de Otimização de Estratégias de Trading
Otimiza parâmetros usando algoritmos genéticos e grid search
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from dataclasses import dataclass
import json
import pickle

from advanced_backtester import AdvancedBacktester, BacktestMetrics
from ml_model import TradingMLModel
from data_collector import data_collector
from utils import get_logger

@dataclass
class OptimizationParameter:
    """Parâmetro para otimização"""
    name: str
    min_value: float
    max_value: float
    step: float
    param_type: str = 'float'  # 'float', 'int', 'choice'
    choices: Optional[List] = None

@dataclass
class OptimizationResult:
    """Resultado da otimização"""
    parameters: Dict[str, Any]
    fitness_score: float
    metrics: BacktestMetrics
    backtest_results: Dict[str, Any]

class StrategyOptimizer:
    """Otimizador de estratégias de trading"""
    
    def __init__(self):
        self.logger = get_logger('optimizer')
        self.backtester = AdvancedBacktester()
        
        self.logger.info("Inicializando Strategy Optimizer...")
        
        # Configurações de otimização
        self.optimization_data = None
        self.base_model = None
        self.optimization_results = []
        self.best_strategy = None
        self.best_performance = 0
        
        # Espaço de busca para parâmetros
        self.parameter_space = {
            'min_confidence': [0.5, 0.6, 0.7, 0.8, 0.9],
            'stake_percentage': [0.01, 0.02, 0.05, 0.1],
            'max_trades_per_day': [5, 10, 20, 50],
            'stop_loss': [0.05, 0.1, 0.15, 0.2],
            'take_profit': [0.1, 0.15, 0.2, 0.3],
            'martingale_steps': [0, 1, 2, 3],
            'timeframe': ['1m', '5m', '15m', '1h']
        }
        
        # Métricas de avaliação
        self.evaluation_metrics = [
            'total_return',
            'sharpe_ratio',
            'max_drawdown',
            'win_rate',
            'profit_factor',
            'avg_trade_duration'
        ]
        
        self.logger.info("Strategy Optimizer inicializado")
        
        # Parâmetros padrão para otimização
        self.default_parameters = [
            OptimizationParameter("min_confidence", 0.5, 0.9, 0.05),
            OptimizationParameter("stake_amount", 0.5, 5.0, 0.5),
            OptimizationParameter("max_trades_per_day", 10, 100, 10, 'int'),
            OptimizationParameter("trade_duration", 1, 20, 1, 'int'),
        ]
    
    def optimize_strategy(self,
                         data: pd.DataFrame,
                         model: TradingMLModel,
                         parameters: Optional[List[OptimizationParameter]] = None,
                         optimization_method: str = 'grid_search',
                         fitness_function: str = 'sharpe_ratio',
                         max_iterations: int = 100,
                         population_size: int = 20,
                         n_jobs: int = 4) -> List[OptimizationResult]:
        """
        Otimiza estratégia de trading
        
        Args:
            data: Dados históricos
            model: Modelo de ML
            parameters: Lista de parâmetros para otimizar
            optimization_method: 'grid_search', 'genetic_algorithm', 'random_search'
            fitness_function: Função de fitness ('sharpe_ratio', 'profit_factor', 'net_profit')
            max_iterations: Máximo de iterações
            population_size: Tamanho da população (para algoritmo genético)
            n_jobs: Número de threads paralelas
        
        Returns:
            Lista de resultados ordenados por fitness
        """
        try:
            self.logger.info(f"🔧 Iniciando otimização com método: {optimization_method}")
            
            # Configurar dados e modelo
            self.optimization_data = data
            self.base_model = model
            
            # Usar parâmetros padrão se não especificados
            if parameters is None:
                parameters = self.default_parameters
            
            # Executar otimização baseada no método
            if optimization_method == 'grid_search':
                results = self._grid_search_optimization(parameters, fitness_function, n_jobs)
            elif optimization_method == 'genetic_algorithm':
                results = self._genetic_algorithm_optimization(
                    parameters, fitness_function, max_iterations, population_size, n_jobs
                )
            elif optimization_method == 'random_search':
                results = self._random_search_optimization(
                    parameters, fitness_function, max_iterations, n_jobs
                )
            else:
                raise ValueError(f"Método de otimização não suportado: {optimization_method}")
            
            # Ordenar por fitness score
            results.sort(key=lambda x: x.fitness_score, reverse=True)
            
            self.optimization_results = results
            
            self.logger.info(f"✅ Otimização concluída! {len(results)} configurações testadas")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Erro na otimização: {e}")
            return []
    
    def _grid_search_optimization(self,
                                 parameters: List[OptimizationParameter],
                                 fitness_function: str,
                                 n_jobs: int) -> List[OptimizationResult]:
        """Otimização por grid search"""
        self.logger.info("Executando Grid Search...")
        
        # Gerar todas as combinações
        param_combinations = self._generate_parameter_combinations(parameters)
        
        self.logger.info(f"Total de combinações: {len(param_combinations)}")
        
        # Executar em paralelo
        results = self._evaluate_parameters_parallel(
            param_combinations, fitness_function, n_jobs
        )
        
        return results
    
    def _genetic_algorithm_optimization(self,
                                      parameters: List[OptimizationParameter],
                                      fitness_function: str,
                                      max_iterations: int,
                                      population_size: int,
                                      n_jobs: int) -> List[OptimizationResult]:
        """Otimização por algoritmo genético"""
        self.logger.info("🧬 Executando Algoritmo Genético...")
        
        # Gerar população inicial
        population = self._generate_random_population(parameters, population_size)
        
        best_results = []
        
        for generation in range(max_iterations):
            self.logger.info(f"🔄 Geração {generation + 1}/{max_iterations}")
            
            # Avaliar população
            generation_results = self._evaluate_parameters_parallel(
                population, fitness_function, n_jobs
            )
            
            # Selecionar melhores
            generation_results.sort(key=lambda x: x.fitness_score, reverse=True)
            best_results.extend(generation_results[:5])  # Manter top 5
            
            # Critério de parada
            if generation > 10:
                recent_scores = [r.fitness_score for r in best_results[-20:]]
                if len(set(recent_scores)) == 1:  # Convergiu
                    self.logger.info("🎯 Algoritmo convergiu!")
                    break
            
            # Gerar nova população
            elite_size = population_size // 4
            elite = [r.parameters for r in generation_results[:elite_size]]
            
            new_population = elite.copy()
            
            # Crossover e mutação
            while len(new_population) < population_size:
                parent1 = random.choice(elite)
                parent2 = random.choice(elite)
                
                child = self._crossover(parent1, parent2, parameters)
                child = self._mutate(child, parameters, mutation_rate=0.1)
                
                new_population.append(child)
            
            population = new_population
        
        return best_results
    
    def _random_search_optimization(self,
                                   parameters: List[OptimizationParameter],
                                   fitness_function: str,
                                   max_iterations: int,
                                   n_jobs: int) -> List[OptimizationResult]:
        """Otimização por busca aleatória"""
        self.logger.info("🎲 Executando Random Search...")
        
        # Gerar combinações aleatórias
        random_combinations = self._generate_random_population(parameters, max_iterations)
        
        # Executar em paralelo
        results = self._evaluate_parameters_parallel(
            random_combinations, fitness_function, n_jobs
        )
        
        return results
    
    def _generate_parameter_combinations(self, parameters: List[OptimizationParameter]) -> List[Dict[str, Any]]:
        """Gera todas as combinações de parâmetros para grid search"""
        param_ranges = []
        
        for param in parameters:
            if param.param_type == 'choice':
                param_ranges.append(param.choices)
            elif param.param_type == 'int':
                param_ranges.append(list(range(int(param.min_value), int(param.max_value) + 1, int(param.step))))
            else:  # float
                values = []
                current = param.min_value
                while current <= param.max_value:
                    values.append(round(current, 3))
                    current += param.step
                param_ranges.append(values)
        
        # Gerar todas as combinações
        combinations = []
        for combo in itertools.product(*param_ranges):
            param_dict = {}
            for i, param in enumerate(parameters):
                param_dict[param.name] = combo[i]
            combinations.append(param_dict)
        
        return combinations
    
    def _generate_random_population(self, parameters: List[OptimizationParameter], size: int) -> List[Dict[str, Any]]:
        """Gera população aleatória"""
        population = []
        
        for _ in range(size):
            individual = {}
            for param in parameters:
                if param.param_type == 'choice':
                    individual[param.name] = random.choice(param.choices)
                elif param.param_type == 'int':
                    individual[param.name] = random.randint(int(param.min_value), int(param.max_value))
                else:  # float
                    individual[param.name] = round(
                        random.uniform(param.min_value, param.max_value), 3
                    )
            population.append(individual)
        
        return population
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any], 
                   parameters: List[OptimizationParameter]) -> Dict[str, Any]:
        """Crossover entre dois indivíduos"""
        child = {}
        
        for param in parameters:
            # Escolher aleatoriamente de qual pai herdar
            if random.random() < 0.5:
                child[param.name] = parent1[param.name]
            else:
                child[param.name] = parent2[param.name]
        
        return child
    
    def _mutate(self, individual: Dict[str, Any], parameters: List[OptimizationParameter], 
                mutation_rate: float) -> Dict[str, Any]:
        """Mutação de um indivíduo"""
        mutated = individual.copy()
        
        for param in parameters:
            if random.random() < mutation_rate:
                if param.param_type == 'choice':
                    mutated[param.name] = random.choice(param.choices)
                elif param.param_type == 'int':
                    mutated[param.name] = random.randint(int(param.min_value), int(param.max_value))
                else:  # float
                    mutated[param.name] = round(
                        random.uniform(param.min_value, param.max_value), 3
                    )
        
        return mutated
    
    def _evaluate_parameters_parallel(self,
                                    parameter_combinations: List[Dict[str, Any]],
                                    fitness_function: str,
                                    n_jobs: int) -> List[OptimizationResult]:
        """Avalia parâmetros em paralelo"""
        results = []
        
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            # Submeter tarefas
            future_to_params = {
                executor.submit(self._evaluate_single_parameter_set, params, fitness_function): params
                for params in parameter_combinations
            }
            
            # Coletar resultados
            for i, future in enumerate(as_completed(future_to_params)):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                    
                    # Log de progresso
                    if (i + 1) % 10 == 0:
                        self.logger.info(f"📊 Progresso: {i + 1}/{len(parameter_combinations)}")
                        
                except Exception as e:
                    params = future_to_params[future]
                    self.logger.warning(f"⚠️ Erro ao avaliar parâmetros {params}: {e}")
        
        return results
    
    def _evaluate_single_parameter_set(self,
                                     parameters: Dict[str, Any],
                                     fitness_function: str) -> Optional[OptimizationResult]:
        """Avalia um conjunto de parâmetros"""
        try:
            # Executar backtesting com os parâmetros
            backtest_results = self.backtester.run_backtest(
                data=self.optimization_data,
                model=self.base_model,
                **parameters
            )
            
            if not backtest_results['success']:
                return None
            
            # Calcular fitness score
            metrics = BacktestMetrics(**backtest_results['metrics'])
            fitness_score = self._calculate_fitness_score(metrics, fitness_function)
            
            return OptimizationResult(
                parameters=parameters,
                fitness_score=fitness_score,
                metrics=metrics,
                backtest_results=backtest_results
            )
            
        except Exception as e:
            self.logger.warning(f"⚠️ Erro na avaliação: {e}")
            return None
    
    def _calculate_fitness_score(self, metrics: BacktestMetrics, fitness_function: str) -> float:
        """Calcula score de fitness"""
        if fitness_function == 'sharpe_ratio':
            return metrics.sharpe_ratio
        elif fitness_function == 'profit_factor':
            return metrics.profit_factor
        elif fitness_function == 'net_profit':
            return metrics.net_profit
        elif fitness_function == 'win_rate':
            return metrics.win_rate
        elif fitness_function == 'expectancy':
            return metrics.expectancy
        elif fitness_function == 'calmar_ratio':
            return metrics.calmar_ratio
        elif fitness_function == 'composite':
            # Score composto
            return (
                metrics.sharpe_ratio * 0.3 +
                metrics.profit_factor * 0.2 +
                metrics.win_rate * 0.01 +  # Converter para escala similar
                metrics.expectancy * 0.3 +
                (100 - metrics.max_drawdown) * 0.01 +  # Penalizar drawdown
                metrics.calmar_ratio * 0.2
            )
        else:
            return metrics.net_profit  # Default
    
    def get_best_parameters(self, top_n: int = 5) -> List[OptimizationResult]:
        """Retorna os melhores parâmetros encontrados"""
        if not self.optimization_results:
            return []
        
        return self.optimization_results[:top_n]
    
    def save_optimization_results(self, filepath: str):
        """Salva resultados da otimização"""
        try:
            results_data = []
            
            for result in self.optimization_results:
                results_data.append({
                    'parameters': result.parameters,
                    'fitness_score': result.fitness_score,
                    'metrics': result.metrics.__dict__,
                    'backtest_summary': {
                        'total_trades': result.metrics.total_trades,
                        'win_rate': result.metrics.win_rate,
                        'net_profit': result.metrics.net_profit,
                        'max_drawdown': result.metrics.max_drawdown
                    }
                })
            
            with open(filepath, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            
            self.logger.info(f"💾 Resultados de otimização salvos em {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao salvar resultados: {e}")
    
    def generate_optimization_report(self) -> str:
        """Gera relatório da otimização"""
        if not self.optimization_results:
            return "Nenhum resultado de otimização disponível"
        
        best_result = self.optimization_results[0]
        
        report = f"""
🔧 RELATÓRIO DE OTIMIZAÇÃO DE ESTRATÉGIA
{'='*60}

📊 RESUMO DA OTIMIZAÇÃO
• Total de configurações testadas: {len(self.optimization_results)}
• Melhor fitness score: {best_result.fitness_score:.4f}

🏆 MELHORES PARÂMETROS
"""
        
        for param, value in best_result.parameters.items():
            report += f"• {param}: {value}\n"
        
        report += f"""
📈 PERFORMANCE DA MELHOR CONFIGURAÇÃO
• Taxa de Acerto: {best_result.metrics.win_rate:.2f}%
• Lucro Líquido: ${best_result.metrics.net_profit:.2f}
• Profit Factor: {best_result.metrics.profit_factor:.2f}
• Sharpe Ratio: {best_result.metrics.sharpe_ratio:.3f}
• Máximo Drawdown: {best_result.metrics.max_drawdown:.2f}%

🔍 TOP 5 CONFIGURAÇÕES
"""
        
        for i, result in enumerate(self.optimization_results[:5], 1):
            report += f"\n{i}. Fitness: {result.fitness_score:.4f} | "
            report += f"Win Rate: {result.metrics.win_rate:.1f}% | "
            report += f"Profit: ${result.metrics.net_profit:.2f}\n"
            
            # Mostrar parâmetros principais
            key_params = ['min_confidence', 'stake_amount', 'max_trades_per_day']
            param_str = " | ".join([f"{k}: {result.parameters.get(k, 'N/A')}" for k in key_params])
            report += f"   {param_str}\n"
        
        report += f"\n{'='*60}"
        
        return report

# Instância global
strategy_optimizer = StrategyOptimizer()

# Funções de conveniência
def optimize_trading_strategy(data: pd.DataFrame, model: TradingMLModel, **kwargs) -> List[OptimizationResult]:
    """Otimiza estratégia de trading"""
    return strategy_optimizer.optimize_strategy(data, model, **kwargs)

def quick_optimization(symbol: str = "R_10", days: int = 30, method: str = 'random_search') -> List[OptimizationResult]:
    """Otimização rápida com dados recentes"""
    try:
        # Coletar dados
        historical_data = data_collector.get_historical_data(symbol, "1m", days * 1440)
        
        if historical_data.empty:
            return []
        
        # Criar modelo
        from ml_model import create_and_train_model
        model = create_and_train_model(historical_data)
        
        # Otimizar
        return strategy_optimizer.optimize_strategy(
            historical_data, model, 
            optimization_method=method,
            max_iterations=50
        )
        
    except Exception as e:
        logging.error(f"Erro na otimização rápida: {e}")
        return []