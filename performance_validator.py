#!/usr/bin/env python3
"""
Validador de Performance dos Modelos
Sistema para validar e comparar performance dos modelos otimizados
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

from utils import setup_logging

logger = logging.getLogger(__name__)

class PerformanceValidator:
    """Validador de performance dos modelos"""
    
    def __init__(self):
        """Inicializa o validador"""
        self.results_dirs = {
            "ensemble": "ensemble_test_results",
            "multi_timeframe": "multi_timeframe_results", 
            "symbol_optimization": "symbol_optimization_results"
        }
        
        self.validation_dir = "performance_validation"
        self.reports_dir = os.path.join(self.validation_dir, "reports")
        self.plots_dir = os.path.join(self.validation_dir, "plots")
        
        for dir_path in [self.validation_dir, self.reports_dir, self.plots_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        self.symbols = ["R_50", "R_100", "R_25", "R_75"]
        
        logger.info("PerformanceValidator inicializado")
    
    def load_results(self) -> Dict:
        """
        Carrega todos os resultados das análises
        
        Returns:
            Dicionário com todos os resultados
        """
        try:
            all_results = {}
            
            for analysis_type, results_dir in self.results_dirs.items():
                if not os.path.exists(results_dir):
                    logger.warning(f"Diretório {results_dir} não encontrado")
                    continue
                
                # Encontrar arquivo mais recente
                files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
                if not files:
                    logger.warning(f"Nenhum arquivo JSON encontrado em {results_dir}")
                    continue
                
                latest_file = max(files, key=lambda f: os.path.getctime(os.path.join(results_dir, f)))
                file_path = os.path.join(results_dir, latest_file)
                
                with open(file_path, 'r') as f:
                    results = json.load(f)
                
                all_results[analysis_type] = results
                logger.info(f"Resultados {analysis_type} carregados: {latest_file}")
            
            return all_results
            
        except Exception as e:
            logger.error(f"Erro ao carregar resultados: {e}")
            return {}
    
    def validate_ensemble_performance(self, ensemble_results: Dict) -> Dict:
        """
        Valida performance dos modelos ensemble
        
        Args:
            ensemble_results: Resultados ensemble
            
        Returns:
            Métricas de validação
        """
        try:
            if 'summary' not in ensemble_results:
                return {}
            
            summary = ensemble_results['summary']
            
            validation_metrics = {
                "model_type": "ensemble",
                "total_tests": summary.get('total_tests', 0),
                "profitable_tests": summary.get('profitable_tests', 0),
                "average_accuracy": summary.get('average_accuracy', 0),
                "average_win_rate": summary.get('average_win_rate', 0),
                "average_roi": summary.get('average_roi', 0),
                "profitability_rate": summary.get('profitable_tests', 0) / max(summary.get('total_tests', 1), 1),
                "consistency_score": self.calculate_consistency_score(ensemble_results),
                "risk_adjusted_return": self.calculate_risk_adjusted_return(ensemble_results)
            }
            
            # Análise por configuração
            config_analysis = self.analyze_configurations(ensemble_results)
            validation_metrics["best_configurations"] = config_analysis
            
            logger.info(f"Ensemble validation - Accuracy: {validation_metrics['average_accuracy']:.2%}, ROI: {validation_metrics['average_roi']:.2%}")
            
            return validation_metrics
            
        except Exception as e:
            logger.error(f"Erro na validação ensemble: {e}")
            return {}
    
    def validate_multi_timeframe_performance(self, mt_results: Dict) -> Dict:
        """
        Valida performance da análise multi-timeframe
        
        Args:
            mt_results: Resultados multi-timeframe
            
        Returns:
            Métricas de validação
        """
        try:
            if 'summary' not in mt_results:
                return {}
            
            summary = mt_results['summary']
            
            validation_metrics = {
                "model_type": "multi_timeframe",
                "symbols_analyzed": summary.get('symbols_analyzed', 0),
                "call_signals": summary.get('call_signals', 0),
                "put_signals": summary.get('put_signals', 0),
                "hold_signals": summary.get('hold_signals', 0),
                "average_confidence": summary.get('average_confidence', 0),
                "correlation_stats": summary.get('correlation_stats', {}),
                "signal_distribution": self.calculate_signal_distribution(mt_results),
                "timeframe_consistency": self.calculate_timeframe_consistency(mt_results)
            }
            
            # Análise de correlações
            correlation_stats = summary.get('correlation_stats', {})
            validation_metrics["correlation_quality"] = {
                "average_correlation": correlation_stats.get('average_correlation', 0),
                "max_correlation": correlation_stats.get('max_correlation', 0),
                "min_correlation": correlation_stats.get('min_correlation', 0),
                "correlation_stability": correlation_stats.get('std_correlation', 0)
            }
            
            logger.info(f"Multi-timeframe validation - Confidence: {validation_metrics['average_confidence']:.2%}")
            
            return validation_metrics
            
        except Exception as e:
            logger.error(f"Erro na validação multi-timeframe: {e}")
            return {}
    
    def validate_symbol_optimization_performance(self, symbol_results: Dict) -> Dict:
        """
        Valida performance da otimização por símbolo
        
        Args:
            symbol_results: Resultados otimização por símbolo
            
        Returns:
            Métricas de validação
        """
        try:
            if 'optimization_results' not in symbol_results:
                return {}
            
            optimization_results = symbol_results['optimization_results']
            
            # Análise por símbolo
            symbol_metrics = {}
            total_accuracy = 0
            successful_optimizations = 0
            
            for symbol, result in optimization_results.items():
                if 'best_accuracy' in result:
                    symbol_metrics[symbol] = {
                        "best_accuracy": result.get('best_accuracy', 0),
                        "best_model": result.get('best_model', 'Unknown'),
                        "optimization_success": result.get('best_accuracy', 0) > 0.6
                    }
                    total_accuracy += result.get('best_accuracy', 0)
                    successful_optimizations += 1
            
            validation_metrics = {
                "model_type": "symbol_optimization",
                "symbols_optimized": len(symbol_metrics),
                "successful_optimizations": successful_optimizations,
                "average_accuracy": total_accuracy / max(successful_optimizations, 1),
                "symbol_metrics": symbol_metrics,
                "model_distribution": self.calculate_model_distribution(optimization_results),
                "optimization_efficiency": successful_optimizations / max(len(optimization_results), 1)
            }
            
            logger.info(f"Symbol optimization validation - Average accuracy: {validation_metrics['average_accuracy']:.2%}")
            
            return validation_metrics
            
        except Exception as e:
            logger.error(f"Erro na validação de otimização por símbolo: {e}")
            return {}
    
    def calculate_consistency_score(self, results: Dict) -> float:
        """
        Calcula score de consistência
        
        Args:
            results: Resultados para análise
            
        Returns:
            Score de consistência (0-1)
        """
        try:
            if 'test_results' not in results:
                return 0.0
            
            test_results = results['test_results']
            accuracies = []
            
            for test in test_results:
                if 'accuracy' in test:
                    accuracies.append(test['accuracy'])
            
            if not accuracies:
                return 0.0
            
            # Consistência baseada no desvio padrão
            mean_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies)
            
            # Score de consistência (menor desvio = maior consistência)
            consistency_score = max(0, 1 - (std_accuracy / max(mean_accuracy, 0.01)))
            
            return consistency_score
            
        except Exception as e:
            logger.error(f"Erro ao calcular consistência: {e}")
            return 0.0
    
    def calculate_risk_adjusted_return(self, results: Dict) -> float:
        """
        Calcula retorno ajustado ao risco
        
        Args:
            results: Resultados para análise
            
        Returns:
            Retorno ajustado ao risco
        """
        try:
            if 'summary' not in results:
                return 0.0
            
            summary = results['summary']
            avg_roi = summary.get('average_roi', 0)
            
            # Simular volatilidade baseada na consistência
            consistency = self.calculate_consistency_score(results)
            volatility = max(0.01, 1 - consistency)  # Maior consistência = menor volatilidade
            
            # Sharpe ratio simplificado
            risk_adjusted_return = avg_roi / volatility
            
            return risk_adjusted_return
            
        except Exception as e:
            logger.error(f"Erro ao calcular retorno ajustado: {e}")
            return 0.0
    
    def analyze_configurations(self, results: Dict) -> List[Dict]:
        """
        Analisa melhores configurações
        
        Args:
            results: Resultados para análise
            
        Returns:
            Lista das melhores configurações
        """
        try:
            if 'test_results' not in results:
                return []
            
            test_results = results['test_results']
            
            # Ordenar por ROI
            sorted_results = sorted(
                test_results,
                key=lambda x: x.get('roi', 0),
                reverse=True
            )
            
            # Top 5 configurações
            best_configs = []
            for i, result in enumerate(sorted_results[:5]):
                config = {
                    "rank": i + 1,
                    "symbol": result.get('symbol', 'Unknown'),
                    "accuracy": result.get('accuracy', 0),
                    "win_rate": result.get('win_rate', 0),
                    "roi": result.get('roi', 0),
                    "total_trades": result.get('total_trades', 0)
                }
                best_configs.append(config)
            
            return best_configs
            
        except Exception as e:
            logger.error(f"Erro na análise de configurações: {e}")
            return []
    
    def calculate_signal_distribution(self, results: Dict) -> Dict:
        """
        Calcula distribuição de sinais
        
        Args:
            results: Resultados para análise
            
        Returns:
            Distribuição de sinais
        """
        try:
            if 'summary' not in results:
                return {}
            
            summary = results['summary']
            
            total_signals = (
                summary.get('call_signals', 0) +
                summary.get('put_signals', 0) +
                summary.get('hold_signals', 0)
            )
            
            if total_signals == 0:
                return {"call_pct": 0, "put_pct": 0, "hold_pct": 0}
            
            distribution = {
                "call_pct": summary.get('call_signals', 0) / total_signals,
                "put_pct": summary.get('put_signals', 0) / total_signals,
                "hold_pct": summary.get('hold_signals', 0) / total_signals
            }
            
            return distribution
            
        except Exception as e:
            logger.error(f"Erro ao calcular distribuição: {e}")
            return {}
    
    def calculate_timeframe_consistency(self, results: Dict) -> float:
        """
        Calcula consistência entre timeframes
        
        Args:
            results: Resultados para análise
            
        Returns:
            Score de consistência entre timeframes
        """
        try:
            # Simular consistência baseada nas correlações
            if 'summary' not in results:
                return 0.0
            
            correlation_stats = results['summary'].get('correlation_stats', {})
            avg_correlation = correlation_stats.get('average_correlation', 0)
            
            # Consistência baseada na correlação média
            consistency = min(avg_correlation, 1.0)
            
            return consistency
            
        except Exception as e:
            logger.error(f"Erro ao calcular consistência de timeframes: {e}")
            return 0.0
    
    def calculate_model_distribution(self, optimization_results: Dict) -> Dict:
        """
        Calcula distribuição de modelos
        
        Args:
            optimization_results: Resultados de otimização
            
        Returns:
            Distribuição de modelos
        """
        try:
            model_counts = {}
            total_models = 0
            
            for symbol, result in optimization_results.items():
                best_model = result.get('best_model', 'Unknown')
                model_counts[best_model] = model_counts.get(best_model, 0) + 1
                total_models += 1
            
            # Converter para percentuais
            model_distribution = {}
            for model, count in model_counts.items():
                model_distribution[model] = count / max(total_models, 1)
            
            return model_distribution
            
        except Exception as e:
            logger.error(f"Erro ao calcular distribuição de modelos: {e}")
            return {}
    
    def create_performance_plots(self, validation_results: Dict):
        """
        Cria gráficos de performance
        
        Args:
            validation_results: Resultados de validação
        """
        try:
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Análise de Performance dos Modelos', fontsize=16, fontweight='bold')
            
            # 1. Comparação de Acurácia
            ax1 = axes[0, 0]
            models = []
            accuracies = []
            
            for model_type, results in validation_results.items():
                if 'average_accuracy' in results:
                    models.append(model_type.replace('_', ' ').title())
                    accuracies.append(results['average_accuracy'])
            
            if models and accuracies:
                bars = ax1.bar(models, accuracies, color=['#2E86AB', '#A23B72', '#F18F01'])
                ax1.set_title('Acurácia Média por Modelo')
                ax1.set_ylabel('Acurácia')
                ax1.set_ylim(0, 1)
                
                # Adicionar valores nas barras
                for bar, acc in zip(bars, accuracies):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{acc:.1%}', ha='center', va='bottom')
            
            # 2. ROI Comparison (apenas para ensemble)
            ax2 = axes[0, 1]
            if 'ensemble' in validation_results and 'average_roi' in validation_results['ensemble']:
                roi_data = [validation_results['ensemble']['average_roi']]
                ax2.bar(['Ensemble'], roi_data, color='#2E86AB')
                ax2.set_title('ROI Médio - Ensemble')
                ax2.set_ylabel('ROI')
                
                # Adicionar valor
                if roi_data:
                    ax2.text(0, roi_data[0] + 0.01, f'{roi_data[0]:.1%}', 
                            ha='center', va='bottom')
            
            # 3. Distribuição de Sinais (Multi-timeframe)
            ax3 = axes[1, 0]
            if 'multi_timeframe' in validation_results and 'signal_distribution' in validation_results['multi_timeframe']:
                signal_dist = validation_results['multi_timeframe']['signal_distribution']
                if signal_dist:
                    labels = ['CALL', 'PUT', 'HOLD']
                    sizes = [signal_dist.get('call_pct', 0), 
                            signal_dist.get('put_pct', 0), 
                            signal_dist.get('hold_pct', 0)]
                    colors = ['#2E86AB', '#A23B72', '#F18F01']
                    
                    ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                    ax3.set_title('Distribuição de Sinais - Multi-timeframe')
            
            # 4. Performance por Símbolo
            ax4 = axes[1, 1]
            if 'symbol_optimization' in validation_results and 'symbol_metrics' in validation_results['symbol_optimization']:
                symbol_metrics = validation_results['symbol_optimization']['symbol_metrics']
                if symbol_metrics:
                    symbols = list(symbol_metrics.keys())
                    symbol_accuracies = [metrics['best_accuracy'] for metrics in symbol_metrics.values()]
                    
                    bars = ax4.bar(symbols, symbol_accuracies, color='#F18F01')
                    ax4.set_title('Acurácia por Símbolo')
                    ax4.set_ylabel('Acurácia')
                    ax4.set_ylim(0, 1)
                    
                    # Adicionar valores
                    for bar, acc in zip(bars, symbol_accuracies):
                        height = bar.get_height()
                        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f'{acc:.1%}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Salvar gráfico
            plot_file = os.path.join(self.plots_dir, f"performance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Gráficos salvos em: {plot_file}")
            
        except Exception as e:
            logger.error(f"Erro ao criar gráficos: {e}")
    
    def generate_validation_report(self, validation_results: Dict) -> str:
        """
        Gera relatório de validação
        
        Args:
            validation_results: Resultados de validação
            
        Returns:
            Caminho do relatório
        """
        try:
            report_file = os.path.join(
                self.reports_dir,
                f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            )
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("# Relatório de Validação de Performance\n\n")
                f.write(f"**Data:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
                
                # Resumo Executivo
                f.write("## 📊 Resumo Executivo\n\n")
                
                total_models = len(validation_results)
                f.write(f"- **Modelos Analisados:** {total_models}\n")
                
                # Melhor modelo por acurácia
                best_accuracy = 0
                best_model = "N/A"
                for model_type, results in validation_results.items():
                    if 'average_accuracy' in results and results['average_accuracy'] > best_accuracy:
                        best_accuracy = results['average_accuracy']
                        best_model = model_type
                
                f.write(f"- **Melhor Modelo (Acurácia):** {best_model.replace('_', ' ').title()} ({best_accuracy:.1%})\n")
                
                # ROI do ensemble
                if 'ensemble' in validation_results and 'average_roi' in validation_results['ensemble']:
                    roi = validation_results['ensemble']['average_roi']
                    f.write(f"- **ROI Ensemble:** {roi:.1%}\n")
                
                f.write("\n---\n\n")
                
                # Análise Detalhada por Modelo
                for model_type, results in validation_results.items():
                    f.write(f"## 🔍 {model_type.replace('_', ' ').title()}\n\n")
                    
                    if model_type == "ensemble":
                        f.write("### Métricas Principais\n")
                        f.write(f"- **Testes Realizados:** {results.get('total_tests', 0)}\n")
                        f.write(f"- **Testes Lucrativos:** {results.get('profitable_tests', 0)}\n")
                        f.write(f"- **Taxa de Lucratividade:** {results.get('profitability_rate', 0):.1%}\n")
                        f.write(f"- **Acurácia Média:** {results.get('average_accuracy', 0):.1%}\n")
                        f.write(f"- **Taxa de Vitória:** {results.get('average_win_rate', 0):.1%}\n")
                        f.write(f"- **ROI Médio:** {results.get('average_roi', 0):.1%}\n")
                        f.write(f"- **Score de Consistência:** {results.get('consistency_score', 0):.1%}\n")
                        f.write(f"- **Retorno Ajustado ao Risco:** {results.get('risk_adjusted_return', 0):.2f}\n\n")
                        
                        # Melhores configurações
                        if 'best_configurations' in results and results['best_configurations']:
                            f.write("### 🏆 Top 5 Configurações\n\n")
                            f.write("| Rank | Símbolo | Acurácia | Taxa Vitória | ROI | Trades |\n")
                            f.write("|------|---------|----------|--------------|-----|--------|\n")
                            
                            for config in results['best_configurations']:
                                f.write(f"| {config['rank']} | {config['symbol']} | {config['accuracy']:.1%} | {config['win_rate']:.1%} | {config['roi']:.1%} | {config['total_trades']} |\n")
                            f.write("\n")
                    
                    elif model_type == "multi_timeframe":
                        f.write("### Métricas Principais\n")
                        f.write(f"- **Símbolos Analisados:** {results.get('symbols_analyzed', 0)}\n")
                        f.write(f"- **Sinais CALL:** {results.get('call_signals', 0)}\n")
                        f.write(f"- **Sinais PUT:** {results.get('put_signals', 0)}\n")
                        f.write(f"- **Sinais HOLD:** {results.get('hold_signals', 0)}\n")
                        f.write(f"- **Confiança Média:** {results.get('average_confidence', 0):.1%}\n")
                        f.write(f"- **Consistência Timeframes:** {results.get('timeframe_consistency', 0):.1%}\n\n")
                        
                        # Qualidade das correlações
                        if 'correlation_quality' in results:
                            corr_quality = results['correlation_quality']
                            f.write("### 📈 Qualidade das Correlações\n")
                            f.write(f"- **Correlação Média:** {corr_quality.get('average_correlation', 0):.1%}\n")
                            f.write(f"- **Correlação Máxima:** {corr_quality.get('max_correlation', 0):.1%}\n")
                            f.write(f"- **Correlação Mínima:** {corr_quality.get('min_correlation', 0):.1%}\n")
                            f.write(f"- **Estabilidade:** {1 - corr_quality.get('correlation_stability', 0):.1%}\n\n")
                    
                    elif model_type == "symbol_optimization":
                        f.write("### Métricas Principais\n")
                        f.write(f"- **Símbolos Otimizados:** {results.get('symbols_optimized', 0)}\n")
                        f.write(f"- **Otimizações Bem-sucedidas:** {results.get('successful_optimizations', 0)}\n")
                        f.write(f"- **Eficiência de Otimização:** {results.get('optimization_efficiency', 0):.1%}\n")
                        f.write(f"- **Acurácia Média:** {results.get('average_accuracy', 0):.1%}\n\n")
                        
                        # Performance por símbolo
                        if 'symbol_metrics' in results and results['symbol_metrics']:
                            f.write("### 📊 Performance por Símbolo\n\n")
                            f.write("| Símbolo | Acurácia | Melhor Modelo | Sucesso |\n")
                            f.write("|---------|----------|---------------|----------|\n")
                            
                            for symbol, metrics in results['symbol_metrics'].items():
                                success = "✅" if metrics['optimization_success'] else "❌"
                                f.write(f"| {symbol} | {metrics['best_accuracy']:.1%} | {metrics['best_model']} | {success} |\n")
                            f.write("\n")
                        
                        # Distribuição de modelos
                        if 'model_distribution' in results and results['model_distribution']:
                            f.write("### 🤖 Distribuição de Modelos\n\n")
                            for model, percentage in results['model_distribution'].items():
                                f.write(f"- **{model}:** {percentage:.1%}\n")
                            f.write("\n")
                    
                    f.write("---\n\n")
                
                # Recomendações
                f.write("## 💡 Recomendações\n\n")
                
                # Recomendação baseada na melhor acurácia
                if best_accuracy > 0.8:
                    f.write(f"✅ **Excelente Performance:** O modelo {best_model.replace('_', ' ').title()} apresenta acurácia superior a 80%.\n\n")
                elif best_accuracy > 0.6:
                    f.write(f"⚠️ **Performance Moderada:** O modelo {best_model.replace('_', ' ').title()} apresenta acurácia moderada. Considere ajustes.\n\n")
                else:
                    f.write(f"❌ **Performance Baixa:** Todos os modelos apresentam acurácia baixa. Revisão necessária.\n\n")
                
                # Recomendações específicas
                if 'ensemble' in validation_results:
                    ensemble_roi = validation_results['ensemble'].get('average_roi', 0)
                    if ensemble_roi > 0.1:
                        f.write("✅ **Ensemble Lucrativo:** ROI superior a 10%. Recomendado para trading.\n\n")
                    elif ensemble_roi > 0:
                        f.write("⚠️ **Ensemble Marginalmente Lucrativo:** ROI positivo mas baixo. Otimização recomendada.\n\n")
                    else:
                        f.write("❌ **Ensemble Não Lucrativo:** ROI negativo. Revisão urgente necessária.\n\n")
                
                f.write("### Próximos Passos\n\n")
                f.write("1. **Otimização de Hiperparâmetros:** Ajustar parâmetros dos modelos com melhor performance\n")
                f.write("2. **Engenharia de Features:** Adicionar novos indicadores técnicos\n")
                f.write("3. **Validação Cruzada:** Implementar validação temporal mais robusta\n")
                f.write("4. **Monitoramento Contínuo:** Acompanhar performance em tempo real\n")
                f.write("5. **Backtesting Estendido:** Testar em períodos mais longos\n\n")
                
                f.write("---\n\n")
                f.write(f"*Relatório gerado automaticamente em {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}*\n")
            
            logger.info(f"Relatório salvo em: {report_file}")
            return report_file
            
        except Exception as e:
            logger.error(f"Erro ao gerar relatório: {e}")
            return ""
    
    def run_validation(self) -> Dict:
        """
        Executa validação completa
        
        Returns:
            Resultados da validação
        """
        try:
            print("\n" + "="*80)
            print("🔍 VALIDAÇÃO DE PERFORMANCE DOS MODELOS")
            print("="*80)
            
            # Carregar resultados
            print("📂 Carregando resultados das análises...")
            all_results = self.load_results()
            
            if not all_results:
                print("❌ Nenhum resultado encontrado para validação")
                return {}
            
            print(f"✅ {len(all_results)} tipos de análise carregados")
            
            # Validar cada tipo de análise
            validation_results = {}
            
            if 'ensemble' in all_results:
                print("🔄 Validando performance ensemble...")
                validation_results['ensemble'] = self.validate_ensemble_performance(all_results['ensemble'])
            
            if 'multi_timeframe' in all_results:
                print("🔄 Validando performance multi-timeframe...")
                validation_results['multi_timeframe'] = self.validate_multi_timeframe_performance(all_results['multi_timeframe'])
            
            if 'symbol_optimization' in all_results:
                print("🔄 Validando performance otimização por símbolo...")
                validation_results['symbol_optimization'] = self.validate_symbol_optimization_performance(all_results['symbol_optimization'])
            
            # Criar gráficos
            print("📊 Gerando gráficos de performance...")
            self.create_performance_plots(validation_results)
            
            # Gerar relatório
            print("📝 Gerando relatório de validação...")
            report_file = self.generate_validation_report(validation_results)
            
            # Salvar resultados da validação
            validation_file = os.path.join(
                self.validation_dir,
                f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            with open(validation_file, 'w') as f:
                json.dump(validation_results, f, indent=2, default=str)
            
            print("\n" + "="*80)
            print("✅ VALIDAÇÃO CONCLUÍDA")
            print("="*80)
            print(f"📊 Modelos validados: {len(validation_results)}")
            print(f"📁 Resultados salvos em: {validation_file}")
            print(f"📝 Relatório salvo em: {report_file}")
            print(f"📈 Gráficos salvos em: {self.plots_dir}")
            print("="*80)
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Erro na validação: {e}")
            print(f"❌ Erro na validação: {e}")
            return {}


def run_performance_validation():
    """Função principal para executar validação de performance"""
    try:
        # Configurar logging
        setup_logging()
        
        # Criar validador
        validator = PerformanceValidator()
        
        # Executar validação
        results = validator.run_validation()
        
        return results
        
    except Exception as e:
        logger.error(f"Erro na validação de performance: {e}")
        print(f"❌ Erro: {e}")
        return {}


if __name__ == "__main__":
    run_performance_validation()