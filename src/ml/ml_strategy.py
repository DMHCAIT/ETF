"""
ML Strategy Integration for ETF Trading System.
Integrates ML predictions with trading strategies for enhanced decision making.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime, timedelta
from enum import Enum

from .prediction_pipeline import MLPredictionPipeline
from ..strategy.base_strategy import BaseStrategy
from ..utils.config import Config

logger = logging.getLogger(__name__)

class MLSignalStrength(Enum):
    """ML signal strength levels."""
    VERY_WEAK = 0.1
    WEAK = 0.3
    MODERATE = 0.5
    STRONG = 0.7
    VERY_STRONG = 0.9

class MLTradingStrategy(BaseStrategy):
    """Trading strategy enhanced with ML predictions."""
    
    def __init__(self, config: Config, ml_pipeline: MLPredictionPipeline):
        """
        Initialize ML-enhanced trading strategy.
        
        Args:
            config: Configuration instance
            ml_pipeline: ML prediction pipeline
        """
        super().__init__(config)
        self.ml_pipeline = ml_pipeline
        
        # ML strategy parameters
        self.ml_weight = 0.6  # Weight of ML signals vs traditional signals
        self.confidence_threshold = 0.6  # Minimum confidence for ML signals
        self.consensus_threshold = 0.7  # Threshold for multi-model consensus
        
        # Risk management
        self.max_ml_position_size = 0.15  # Maximum position size for ML signals
        self.stop_loss_ml = 0.02  # Stop loss for ML-based trades
        self.take_profit_ml = 0.04  # Take profit for ML-based trades
        
        # Performance tracking
        self.ml_trade_history = []
        self.ml_performance_metrics = {}
        
        logger.info("ML Trading Strategy initialized")
    
    async def analyze_symbol(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze symbol with ML-enhanced signals.
        
        Args:
            symbol: ETF symbol
            data: Market data
            
        Returns:
            Analysis results with ML predictions
        """
        try:
            # Get traditional technical analysis
            traditional_analysis = await super().analyze_symbol(symbol, data)
            
            # Get ML predictions
            ml_predictions = await self.ml_pipeline.get_current_predictions(symbol)
            
            # Combine traditional and ML analysis
            enhanced_analysis = await self._combine_analyses(
                traditional_analysis, ml_predictions, symbol
            )
            
            return enhanced_analysis
            
        except Exception as e:
            logger.error(f"Error in ML analysis for {symbol}: {e}")
            return traditional_analysis if 'traditional_analysis' in locals() else {}
    
    async def _combine_analyses(self, traditional: Dict[str, Any], 
                               ml_predictions: Dict[str, Any], 
                               symbol: str) -> Dict[str, Any]:
        """Combine traditional and ML analysis results."""
        try:
            combined_analysis = traditional.copy()
            
            if not ml_predictions:
                combined_analysis['ml_status'] = 'no_predictions'
                return combined_analysis
            
            # Extract ML signals
            ml_signals = self._extract_ml_signals(ml_predictions)
            
            # Calculate combined signal strength
            combined_signal = self._calculate_combined_signal(traditional, ml_signals)
            
            # Update analysis with ML insights
            combined_analysis.update({
                'ml_predictions': ml_predictions,
                'ml_signals': ml_signals,
                'combined_signal': combined_signal,
                'ml_confidence': ml_predictions.get('confidence', 0.0),
                'ml_status': 'active',
                'recommendation': self._generate_ml_recommendation(combined_signal, ml_predictions)
            })
            
            return combined_analysis
            
        except Exception as e:
            logger.error(f"Error combining analyses: {e}")
            return traditional
    
    def _extract_ml_signals(self, ml_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Extract trading signals from ML predictions."""
        signals = {
            'direction_1m': 0.0,
            'direction_5m': 0.0,
            'price_momentum': 0.0,
            'volatility_signal': 0.0,
            'overall_strength': 0.0
        }
        
        try:
            # Short-term direction signal
            if 'direction_1m_prob' in ml_predictions:
                prob = ml_predictions['direction_1m_prob']
                # Convert probability to signal strength (-1 to 1)
                signals['direction_1m'] = (prob - 0.5) * 2
            
            # Medium-term direction signal
            if 'direction_5m_prob' in ml_predictions:
                prob = ml_predictions['direction_5m_prob']
                signals['direction_5m'] = (prob - 0.5) * 2
            
            # Price momentum signal
            if 'price_change_1m' in ml_predictions:
                change = ml_predictions['price_change_1m']
                # Normalize to [-1, 1] range
                signals['price_momentum'] = np.tanh(change * 100)  # Scale by 100 for sensitivity
            
            # Volatility signal (higher volatility = higher uncertainty)
            if 'volatility_forecast' in ml_predictions:
                vol = ml_predictions['volatility_forecast']
                # Lower volatility = more confident signal
                signals['volatility_signal'] = 1 / (1 + vol * 100)
            
            # Overall signal strength
            confidence = ml_predictions.get('confidence', 0.5)
            direction_consistency = abs(signals['direction_1m'] - signals['direction_5m'])
            
            signals['overall_strength'] = confidence * (1 - direction_consistency * 0.5)
            
        except Exception as e:
            logger.error(f"Error extracting ML signals: {e}")
        
        return signals
    
    def _calculate_combined_signal(self, traditional: Dict[str, Any], 
                                  ml_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate combined signal from traditional and ML analysis."""
        try:
            # Get traditional signal strength
            traditional_strength = traditional.get('signal_strength', 0.0)
            traditional_direction = 1 if traditional.get('signal', 'hold').lower() == 'buy' else -1 if traditional.get('signal', 'hold').lower() == 'sell' else 0
            
            # Get ML signal strength
            ml_strength = ml_signals['overall_strength']
            ml_direction = np.sign(ml_signals['direction_1m'] + ml_signals['direction_5m'])
            
            # Weighted combination
            traditional_weight = 1 - self.ml_weight
            ml_weight = self.ml_weight
            
            # Combined direction
            combined_direction_score = (
                traditional_direction * traditional_strength * traditional_weight +
                ml_direction * ml_strength * ml_weight
            )
            
            # Combined strength
            combined_strength = (
                traditional_strength * traditional_weight +
                ml_strength * ml_weight
            )
            
            # Generate signal
            if abs(combined_direction_score) < 0.1:
                signal = 'hold'
            elif combined_direction_score > 0:
                signal = 'buy'
            else:
                signal = 'sell'
            
            return {
                'signal': signal,
                'strength': combined_strength,
                'direction_score': combined_direction_score,
                'traditional_component': traditional_strength * traditional_weight,
                'ml_component': ml_strength * ml_weight,
                'consensus': self._check_consensus(traditional_direction, ml_direction)
            }
            
        except Exception as e:
            logger.error(f"Error calculating combined signal: {e}")
            return {'signal': 'hold', 'strength': 0.0}
    
    def _check_consensus(self, traditional_direction: float, ml_direction: float) -> bool:
        """Check if traditional and ML signals agree."""
        if traditional_direction == 0 or ml_direction == 0:
            return False
        return np.sign(traditional_direction) == np.sign(ml_direction)
    
    def _generate_ml_recommendation(self, combined_signal: Dict[str, Any], 
                                  ml_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed trading recommendation with ML insights."""
        try:
            signal = combined_signal['signal']
            strength = combined_signal['strength']
            confidence = ml_predictions.get('confidence', 0.0)
            
            # Determine recommendation strength
            if strength >= MLSignalStrength.VERY_STRONG.value:
                rec_strength = 'Very Strong'
            elif strength >= MLSignalStrength.STRONG.value:
                rec_strength = 'Strong'
            elif strength >= MLSignalStrength.MODERATE.value:
                rec_strength = 'Moderate'
            elif strength >= MLSignalStrength.WEAK.value:
                rec_strength = 'Weak'
            else:
                rec_strength = 'Very Weak'
            
            # Position sizing based on ML confidence
            if signal != 'hold':
                position_size = min(
                    self.max_ml_position_size * confidence * strength,
                    self.max_ml_position_size
                )
            else:
                position_size = 0.0
            
            # Risk management levels
            if signal == 'buy':
                stop_loss = -self.stop_loss_ml
                take_profit = self.take_profit_ml
            elif signal == 'sell':
                stop_loss = self.stop_loss_ml
                take_profit = -self.take_profit_ml
            else:
                stop_loss = 0.0
                take_profit = 0.0
            
            recommendation = {
                'action': signal,
                'strength': rec_strength,
                'confidence': confidence,
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'holding_period': self._estimate_holding_period(ml_predictions),
                'risk_level': self._assess_risk_level(ml_predictions, combined_signal),
                'rationale': self._generate_rationale(ml_predictions, combined_signal)
            }
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error generating ML recommendation: {e}")
            return {'action': 'hold', 'strength': 'Unknown'}
    
    def _estimate_holding_period(self, ml_predictions: Dict[str, Any]) -> str:
        """Estimate optimal holding period based on ML predictions."""
        try:
            # Check consistency between timeframes
            direction_1m = ml_predictions.get('direction_1m_prob', 0.5)
            direction_5m = ml_predictions.get('direction_5m_prob', 0.5)
            
            consistency = 1 - abs(direction_1m - direction_5m)
            
            if consistency > 0.8:
                return 'medium_term'  # 5-15 minutes
            elif consistency > 0.6:
                return 'short_term'   # 1-5 minutes
            else:
                return 'very_short'   # < 1 minute
                
        except:
            return 'short_term'
    
    def _assess_risk_level(self, ml_predictions: Dict[str, Any], 
                          combined_signal: Dict[str, Any]) -> str:
        """Assess risk level of the trade."""
        try:
            confidence = ml_predictions.get('confidence', 0.0)
            volatility = ml_predictions.get('volatility_forecast', 0.01)
            consensus = combined_signal.get('consensus', False)
            
            # Calculate risk score
            risk_score = 0.0
            
            # Lower confidence = higher risk
            risk_score += (1 - confidence) * 0.4
            
            # Higher volatility = higher risk
            risk_score += min(volatility * 100, 1.0) * 0.3
            
            # No consensus = higher risk
            if not consensus:
                risk_score += 0.3
            
            if risk_score < 0.3:
                return 'low'
            elif risk_score < 0.6:
                return 'medium'
            else:
                return 'high'
                
        except:
            return 'medium'
    
    def _generate_rationale(self, ml_predictions: Dict[str, Any], 
                           combined_signal: Dict[str, Any]) -> str:
        """Generate human-readable rationale for the recommendation."""
        try:
            rationale_parts = []
            
            # ML confidence
            confidence = ml_predictions.get('confidence', 0.0)
            if confidence > 0.7:
                rationale_parts.append("High ML model confidence")
            elif confidence > 0.5:
                rationale_parts.append("Moderate ML model confidence")
            else:
                rationale_parts.append("Low ML model confidence")
            
            # Direction consistency
            direction_1m = ml_predictions.get('direction_1m_prob', 0.5)
            direction_5m = ml_predictions.get('direction_5m_prob', 0.5)
            
            if abs(direction_1m - direction_5m) < 0.2:
                rationale_parts.append("consistent directional signals across timeframes")
            else:
                rationale_parts.append("mixed signals across timeframes")
            
            # Traditional vs ML consensus
            if combined_signal.get('consensus', False):
                rationale_parts.append("traditional and ML analysis agree")
            else:
                rationale_parts.append("traditional and ML analysis diverge")
            
            # Volatility
            volatility = ml_predictions.get('volatility_forecast', 0.01)
            if volatility > 0.02:
                rationale_parts.append("elevated volatility expected")
            else:
                rationale_parts.append("low volatility environment")
            
            return "; ".join(rationale_parts).capitalize()
            
        except Exception as e:
            logger.error(f"Error generating rationale: {e}")
            return "ML-enhanced analysis with traditional indicators"
    
    async def should_enter_trade(self, symbol: str, analysis: Dict[str, Any]) -> bool:
        """Determine if should enter trade based on ML-enhanced analysis."""
        try:
            # Check if ML predictions are available
            if analysis.get('ml_status') != 'active':
                return await super().should_enter_trade(symbol, analysis)
            
            ml_confidence = analysis.get('ml_confidence', 0.0)
            combined_signal = analysis.get('combined_signal', {})
            recommendation = analysis.get('recommendation', {})
            
            # Entry criteria
            entry_conditions = [
                # ML confidence above threshold
                ml_confidence >= self.confidence_threshold,
                
                # Strong combined signal
                combined_signal.get('strength', 0.0) >= MLSignalStrength.MODERATE.value,
                
                # Clear directional signal
                combined_signal.get('signal', 'hold') != 'hold',
                
                # Risk level acceptable
                recommendation.get('risk_level', 'high') != 'high'
            ]
            
            # Consensus bonus (not required but preferred)
            if combined_signal.get('consensus', False):
                # Lower the strength requirement if we have consensus
                entry_conditions[1] = combined_signal.get('strength', 0.0) >= MLSignalStrength.WEAK.value
            
            return all(entry_conditions)
            
        except Exception as e:
            logger.error(f"Error determining trade entry for {symbol}: {e}")
            return False
    
    async def should_exit_trade(self, symbol: str, position: Dict[str, Any], 
                               analysis: Dict[str, Any]) -> bool:
        """Determine if should exit trade based on ML-enhanced analysis."""
        try:
            # Check traditional exit conditions first
            traditional_exit = await super().should_exit_trade(symbol, position, analysis)
            
            if traditional_exit:
                return True
            
            # ML-specific exit conditions
            if analysis.get('ml_status') != 'active':
                return False
            
            ml_predictions = analysis.get('ml_predictions', {})
            combined_signal = analysis.get('combined_signal', {})
            
            # Current position details
            position_side = position.get('side', 'long')
            entry_price = position.get('entry_price', 0.0)
            current_price = position.get('current_price', entry_price)
            
            # Calculate current P&L
            if position_side == 'long':
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price
            
            # ML exit conditions
            ml_exit_conditions = [
                # Signal reversal
                self._check_signal_reversal(position_side, combined_signal),
                
                # Confidence drop
                ml_predictions.get('confidence', 1.0) < 0.3,
                
                # Take profit hit
                pnl_pct >= self.take_profit_ml,
                
                # Stop loss hit
                pnl_pct <= -self.stop_loss_ml
            ]
            
            return any(ml_exit_conditions)
            
        except Exception as e:
            logger.error(f"Error determining trade exit for {symbol}: {e}")
            return False
    
    def _check_signal_reversal(self, position_side: str, combined_signal: Dict[str, Any]) -> bool:
        """Check if the current signal contradicts the position."""
        try:
            current_signal = combined_signal.get('signal', 'hold')
            signal_strength = combined_signal.get('strength', 0.0)
            
            # Strong reversal signal
            if signal_strength >= MLSignalStrength.MODERATE.value:
                if position_side == 'long' and current_signal == 'sell':
                    return True
                elif position_side == 'short' and current_signal == 'buy':
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking signal reversal: {e}")
            return False
    
    def calculate_position_size(self, symbol: str, analysis: Dict[str, Any], 
                              available_capital: float) -> float:
        """Calculate position size based on ML confidence and risk assessment."""
        try:
            # Get base position size from traditional strategy
            base_size = super().calculate_position_size(symbol, analysis, available_capital)
            
            # Adjust based on ML insights
            if analysis.get('ml_status') != 'active':
                return base_size
            
            recommendation = analysis.get('recommendation', {})
            ml_confidence = analysis.get('ml_confidence', 0.0)
            risk_level = recommendation.get('risk_level', 'medium')
            
            # Risk multipliers
            risk_multipliers = {
                'low': 1.2,
                'medium': 1.0,
                'high': 0.7
            }
            
            # Confidence multiplier
            confidence_multiplier = 0.5 + (ml_confidence * 0.8)  # Range: 0.5 to 1.3
            
            # Calculate adjusted size
            adjusted_size = base_size * risk_multipliers[risk_level] * confidence_multiplier
            
            # Cap at maximum ML position size
            max_ml_size = available_capital * self.max_ml_position_size
            
            return min(adjusted_size, max_ml_size)
            
        except Exception as e:
            logger.error(f"Error calculating ML position size: {e}")
            return super().calculate_position_size(symbol, analysis, available_capital)
    
    def get_ml_performance_summary(self) -> Dict[str, Any]:
        """Get summary of ML strategy performance."""
        try:
            if not self.ml_trade_history:
                return {'status': 'no_trades'}
            
            # Calculate performance metrics
            total_trades = len(self.ml_trade_history)
            winning_trades = sum(1 for trade in self.ml_trade_history if trade.get('pnl', 0) > 0)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            total_pnl = sum(trade.get('pnl', 0) for trade in self.ml_trade_history)
            avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
            
            # ML-specific metrics
            high_confidence_trades = sum(1 for trade in self.ml_trade_history 
                                       if trade.get('ml_confidence', 0) > 0.7)
            consensus_trades = sum(1 for trade in self.ml_trade_history 
                                 if trade.get('consensus', False))
            
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl,
                'high_confidence_trades': high_confidence_trades,
                'consensus_trades': consensus_trades,
                'ml_weight': self.ml_weight,
                'confidence_threshold': self.confidence_threshold
            }
            
        except Exception as e:
            logger.error(f"Error getting ML performance summary: {e}")
            return {'status': 'error'}
    
    async def update_ml_parameters(self, new_params: Dict[str, Any]) -> bool:
        """Update ML strategy parameters based on performance."""
        try:
            if 'ml_weight' in new_params:
                self.ml_weight = max(0.0, min(1.0, new_params['ml_weight']))
            
            if 'confidence_threshold' in new_params:
                self.confidence_threshold = max(0.0, min(1.0, new_params['confidence_threshold']))
            
            if 'consensus_threshold' in new_params:
                self.consensus_threshold = max(0.0, min(1.0, new_params['consensus_threshold']))
            
            if 'max_ml_position_size' in new_params:
                self.max_ml_position_size = max(0.01, min(0.5, new_params['max_ml_position_size']))
            
            logger.info("Updated ML strategy parameters")
            return True
            
        except Exception as e:
            logger.error(f"Error updating ML parameters: {e}")
            return False
