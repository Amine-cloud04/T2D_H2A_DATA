import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta
from loguru import logger

class KPICalculator:
    """Calcule tous les KPIs définis dans le cahier des charges"""
    
    def __init__(self):
        self.kpis = {}
    
    def calculate_all_kpis(self, df: pd.DataFrame, sentiments: List[Dict] = None) -> Dict:
        """Calcule tous les KPIs"""
        logger.info("Calculating all KPIs...")
        
        kpis = {
            'global': self._calculate_global_kpis(df),
            'criteria': self._calculate_criteria_kpis(df),
            'sentiment': self._calculate_sentiment_kpis(sentiments) if sentiments else {},
            'temporal': self._calculate_temporal_kpis(df),
            'formation_type': self._calculate_formation_type_kpis(df),
            'performance': self._calculate_performance_kpis(df)
        }
        
        logger.info("All KPIs calculated")
        return kpis
    
    def _calculate_global_kpis(self, df: pd.DataFrame) -> Dict:
        """KPIs globaux"""
        total_evaluations = len(df)
        
        # Taux de complétion (supposons que les évaluations complètes ont tous les champs)
        complete_evaluations = df.dropna(subset=['satisfaction', 'contenu', 'logistique', 'applicabilite'])
        completion_rate = (len(complete_evaluations) / total_evaluations * 100) if total_evaluations > 0 else 0
        
        return {
            'total_evaluations': total_evaluations,
            'completion_rate': round(completion_rate, 1),
            'avg_satisfaction': round(df['satisfaction'].mean(), 2),
            'median_satisfaction': round(df['satisfaction'].median(), 2),
            'std_satisfaction': round(df['satisfaction'].std(), 2),
            'date_debut': df['date'].min().strftime('%Y-%m-%d') if not df.empty else None,
            'date_fin': df['date'].max().strftime('%Y-%m-%d') if not df.empty else None
        }
    
    def _calculate_criteria_kpis(self, df: pd.DataFrame) -> Dict:
        """KPIs par critère d'évaluation"""
        criteria = ['satisfaction', 'contenu', 'logistique', 'applicabilite']
        
        result = {}
        for criterion in criteria:
            if criterion in df.columns:
                result[criterion] = {
                    'mean': round(df[criterion].mean(), 2),
                    'median': round(df[criterion].median(), 2),
                    'std': round(df[criterion].std(), 2),
                    'min': int(df[criterion].min()),
                    'max': int(df[criterion].max()),
                    'distribution': df[criterion].value_counts().to_dict()
                }
        
        return result
    
    def _calculate_sentiment_kpis(self, sentiments: List[Dict]) -> Dict:
        """KPIs d'analyse de sentiment"""
        if not sentiments:
            return {}
        
        total = len(sentiments)
        
        positive = sum(1 for s in sentiments if s['label'] == 'POSITIVE')
        neutral = sum(1 for s in sentiments if s['label'] == 'NEUTRAL')
        negative = sum(1 for s in sentiments if s['label'] == 'NEGATIVE')
        
        avg_confidence = np.mean([s['confidence'] for s in sentiments])
        
        return {
            'total_analyzed': total,
            'positive_count': positive,
            'neutral_count': neutral,
            'negative_count': negative,
            'positive_pct': round((positive / total * 100), 1) if total > 0 else 0,
            'neutral_pct': round((neutral / total * 100), 1) if total > 0 else 0,
            'negative_pct': round((negative / total * 100), 1) if total > 0 else 0,
            'avg_confidence': round(avg_confidence, 3)
        }
    
    def _calculate_temporal_kpis(self, df: pd.DataFrame) -> Dict:
        """KPIs d'évolution temporelle"""
        if 'date' not in df.columns or df.empty:
            return{}
    
        try:
            # Convertir en datetime
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
        
            # Grouper par mois (créer une colonne string directement)
            df['year_month_str'] = df['date'].dt.strftime('%Y-%m')
            
            monthly_stats = df.groupby('year_month_str').agg({
                'satisfaction': ['mean', 'count'],
                'contenu': 'mean',
                'logistique': 'mean',
                'applicabilite': 'mean'
            }).reset_index()
            
            monthly_stats.columns = ['month', 'satisfaction_mean', 'count', 'contenu_mean', 
                                    'logistique_mean', 'applicabilite_mean']
            
            # Arrondir les valeurs
            for col in ['satisfaction_mean', 'contenu_mean', 'logistique_mean', 'applicabilite_mean']:
                monthly_stats[col] = monthly_stats[col].round(2)
            
            # Calculer la tendance
            if len(monthly_stats) > 1:
                months_numeric = list(range(len(monthly_stats)))
                satisfaction_values = monthly_stats['satisfaction_mean'].values
                
                z = np.polyfit(months_numeric, satisfaction_values, 1)
                trend_slope = float(z[0])
                
                if satisfaction_values[0] != 0:
                        evolution_pct = float((satisfaction_values[-1] - satisfaction_values[0]) / 
                                            satisfaction_values[0] * 100)
                else:
                        evolution_pct = 0.0
            else:
                trend_slope = 0.0
                evolution_pct = 0.0
            
            # Convertir le DataFrame en dict avec des types Python natifs
            monthly_data = []
            for _, row in monthly_stats.iterrows():
                monthly_data.append({
                    'month': str(row['month']),
                    'satisfaction_mean': float(row['satisfaction_mean']),
                    'count': int(row['count']),
                    'contenu_mean': float(row['contenu_mean']),
                    'logistique_mean': float(row['logistique_mean']),
                    'applicabilite_mean': float(row['applicabilite_mean'])
                })
            
            return {
                'monthly_data': monthly_data,
                'trend_slope': round(trend_slope, 4),
                'evolution_pct': round(evolution_pct, 1),
                'trend_direction': 'hausse' if trend_slope > 0 else 'baisse' if trend_slope < 0 else 'stable'
            }
        
        except Exception as e:
            logger.error(f"Error in temporal KPIs: {e}")
            return {}
    
    def _calculate_formation_type_kpis(self, df: pd.DataFrame) -> Dict:
        """KPIs par type de formation"""
        if 'type_formation' not in df.columns:
            return {}
        
        formation_stats = df.groupby('type_formation').agg({
            'satisfaction': ['mean', 'std', 'count'],
            'contenu': 'mean',
            'logistique': 'mean',
            'applicabilite': 'mean',
            'formation_id': 'nunique'
        }).reset_index()
        
        formation_stats.columns = ['type', 'satisfaction_mean', 'satisfaction_std', 'count',
                                    'contenu_mean', 'logistique_mean', 'applicabilite_mean', 
                                    'nb_formations_uniques']
        
        # Trier par satisfaction
        formation_stats = formation_stats.sort_values('satisfaction_mean', ascending=False)
        
        return {
            'by_type': formation_stats.to_dict('records'),
            'best_type': formation_stats.iloc[0]['type'] if not formation_stats.empty else None,
            'worst_type': formation_stats.iloc[-1]['type'] if not formation_stats.empty else None
        }
    
    def _calculate_performance_kpis(self, df: pd.DataFrame) -> Dict:
        """KPIs de performance système"""
        # Simuler le temps de traitement (en production, mesurer réellement)
        avg_processing_time = 1.8  # secondes par évaluation
        
        return {
            'avg_processing_time_seconds': avg_processing_time,
            'total_processing_time_estimate': round(len(df) * avg_processing_time, 1),
            'throughput_per_hour': round(3600 / avg_processing_time, 0)
        }
    
    def compare_periods(self, df: pd.DataFrame, period1_start, period1_end, 
                       period2_start, period2_end) -> Dict:
        """Compare deux périodes"""
        df['date'] = pd.to_datetime(df['date'])
        
        period1 = df[(df['date'] >= period1_start) & (df['date'] <= period1_end)]
        period2 = df[(df['date'] >= period2_start) & (df['date'] <= period2_end)]
        
        comparison = {
            'period1': {
                'count': len(period1),
                'avg_satisfaction': round(period1['satisfaction'].mean(), 2) if not period1.empty else 0
            },
            'period2': {
                'count': len(period2),
                'avg_satisfaction': round(period2['satisfaction'].mean(), 2) if not period2.empty else 0
            }
        }
        
        # Calculer la différence
        if comparison['period1']['avg_satisfaction'] > 0:
            diff_pct = ((comparison['period2']['avg_satisfaction'] - 
                        comparison['period1']['avg_satisfaction']) / 
                       comparison['period1']['avg_satisfaction'] * 100)
            comparison['evolution_pct'] = round(diff_pct, 1)
        else:
            comparison['evolution_pct'] = 0
        
        return comparison
