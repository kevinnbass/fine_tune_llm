"""Disagreement miner for active learning."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
from scipy.stats import entropy
import logging

logger = logging.getLogger(__name__)


class DisagreementMiner:
    """Mine disagreements and interesting samples for active learning."""
    
    def __init__(
        self,
        budget_per_week: int = 100,
        priority_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize disagreement miner.
        
        Args:
            budget_per_week: Number of samples to review per week
            priority_weights: Weights for different disagreement signals
        """
        self.budget_per_week = budget_per_week
        self.priority_weights = priority_weights or {
            'entropy': 1.0,
            'margin': 1.0,
            'vote_split': 1.5,
            'rare_slice': 2.0,
            'abstention': 1.2,
            'model_disagreement': 1.8
        }
        
    def compute_disagreement_scores(
        self,
        voter_outputs_list: List[Dict[str, Dict]],
        texts: List[str],
        metadata_list: Optional[List[Dict]] = None,
        slice_membership: Optional[Dict[str, List[bool]]] = None
    ) -> pd.DataFrame:
        """
        Compute disagreement scores for all samples.
        
        Args:
            voter_outputs_list: List of voter outputs for each sample
            texts: List of input texts
            metadata_list: Optional metadata for each sample
            slice_membership: Dict mapping slice names to membership lists
            
        Returns:
            DataFrame with disagreement scores
        """
        scores_data = []
        
        for idx, (voter_outputs, text) in enumerate(zip(voter_outputs_list, texts)):
            metadata = metadata_list[idx] if metadata_list else {}
            
            # Compute various disagreement signals
            signals = self._compute_signals(voter_outputs, text, metadata)
            
            # Add slice information
            if slice_membership:
                slice_names = [name for name, members in slice_membership.items() 
                             if members[idx]]
                signals['slices'] = ','.join(slice_names) if slice_names else 'none'
                signals['n_slices'] = len(slice_names)
                
                # Check for rare slices
                rare_slice_score = 0.0
                for slice_name in slice_names:
                    slice_size = sum(slice_membership[slice_name])
                    if slice_size < 100:  # Rare slice threshold
                        rare_slice_score = max(rare_slice_score, 1.0 / (slice_size + 1))
                signals['rare_slice_score'] = rare_slice_score
            else:
                signals['slices'] = 'none'
                signals['n_slices'] = 0
                signals['rare_slice_score'] = 0.0
            
            # Compute overall priority score
            priority_score = (
                self.priority_weights['entropy'] * signals['entropy'] +
                self.priority_weights['margin'] * (1 - signals['margin']) +
                self.priority_weights['vote_split'] * signals['vote_split'] +
                self.priority_weights['rare_slice'] * signals['rare_slice_score'] +
                self.priority_weights['abstention'] * signals['abstention_rate'] +
                self.priority_weights['model_disagreement'] * signals['model_disagreement']
            )
            
            signals['priority_score'] = priority_score
            signals['idx'] = idx
            signals['text_preview'] = text[:100] + '...' if len(text) > 100 else text
            
            scores_data.append(signals)
        
        df = pd.DataFrame(scores_data)
        df = df.sort_values('priority_score', ascending=False)
        
        return df
    
    def _compute_signals(
        self,
        voter_outputs: Dict[str, Dict],
        text: str,
        metadata: Dict
    ) -> Dict[str, float]:
        """Compute disagreement signals for a single sample."""
        signals = {}
        
        # Collect predictions and probabilities
        predictions = []
        all_probs = []
        abstentions = 0
        voter_types = {'regex': False, 'classical': False, 'llm': False}
        
        for voter_id, output in voter_outputs.items():
            if output.get('abstain'):
                abstentions += 1
            else:
                probs = output.get('probs', {})
                if probs:
                    pred = max(probs, key=probs.get)
                    predictions.append(pred)
                    all_probs.append(list(probs.values()))
                    
                    # Track voter types
                    if 'regex' in voter_id:
                        voter_types['regex'] = pred
                    elif 'lr' in voter_id or 'svm' in voter_id:
                        voter_types['classical'] = pred
                    elif 'llm' in voter_id:
                        voter_types['llm'] = pred
        
        # Entropy (uncertainty)
        if all_probs:
            mean_probs = np.mean(all_probs, axis=0)
            signals['entropy'] = entropy(mean_probs)
        else:
            signals['entropy'] = 1.0  # Max uncertainty if no predictions
        
        # Margin (confidence)
        if all_probs:
            mean_probs_sorted = sorted(mean_probs, reverse=True)
            margin = mean_probs_sorted[0] - mean_probs_sorted[1] if len(mean_probs_sorted) > 1 else 1.0
            signals['margin'] = margin
        else:
            signals['margin'] = 0.0
        
        # Vote split
        if predictions:
            pred_counts = Counter(predictions)
            most_common = pred_counts.most_common(1)[0][1]
            signals['vote_split'] = 1.0 - (most_common / len(predictions))
        else:
            signals['vote_split'] = 1.0
        
        # Abstention rate
        signals['abstention_rate'] = abstentions / len(voter_outputs) if voter_outputs else 0.0
        
        # Model type disagreement
        model_disagreement = 0.0
        voter_preds = [v for v in voter_types.values() if v]
        if len(voter_preds) > 1 and len(set(voter_preds)) > 1:
            model_disagreement = 1.0
        signals['model_disagreement'] = model_disagreement
        
        # Prediction distribution
        if predictions:
            pred_dist = Counter(predictions)
            for label in ['HIGH_RISK', 'MEDIUM_RISK', 'LOW_RISK', 'NO_RISK']:
                signals[f'pred_{label}'] = pred_dist.get(label, 0) / len(predictions)
        else:
            for label in ['HIGH_RISK', 'MEDIUM_RISK', 'LOW_RISK', 'NO_RISK']:
                signals[f'pred_{label}'] = 0.0
        
        # Text characteristics
        signals['text_length'] = len(text)
        signals['n_voters'] = len(voter_outputs)
        signals['n_predictions'] = len(predictions)
        
        return signals
    
    def select_for_review(
        self,
        disagreement_df: pd.DataFrame,
        strategy: str = 'priority',
        diversity_bonus: bool = True
    ) -> pd.DataFrame:
        """
        Select samples for review based on strategy.
        
        Args:
            disagreement_df: DataFrame with disagreement scores
            strategy: Selection strategy ('priority', 'diverse', 'balanced')
            diversity_bonus: Whether to add diversity bonus
            
        Returns:
            Selected samples DataFrame
        """
        if strategy == 'priority':
            # Simple priority-based selection
            selected = disagreement_df.head(self.budget_per_week)
            
        elif strategy == 'diverse':
            # Select diverse samples across different disagreement types
            selected_indices = []
            
            # High entropy samples
            high_entropy = disagreement_df.nlargest(self.budget_per_week // 4, 'entropy')
            selected_indices.extend(high_entropy['idx'].tolist())
            
            # Low margin samples
            low_margin = disagreement_df.nsmallest(self.budget_per_week // 4, 'margin')
            selected_indices.extend(low_margin['idx'].tolist())
            
            # High abstention samples
            high_abstention = disagreement_df.nlargest(self.budget_per_week // 4, 'abstention_rate')
            selected_indices.extend(high_abstention['idx'].tolist())
            
            # Model disagreement samples
            model_disagree = disagreement_df[disagreement_df['model_disagreement'] > 0.5]
            if len(model_disagree) > 0:
                sample_size = min(len(model_disagree), self.budget_per_week // 4)
                selected_indices.extend(model_disagree.head(sample_size)['idx'].tolist())
            
            # Remove duplicates and get final selection
            selected_indices = list(dict.fromkeys(selected_indices))[:self.budget_per_week]
            selected = disagreement_df[disagreement_df['idx'].isin(selected_indices)]
            
        elif strategy == 'balanced':
            # Balance across predicted classes
            selected_dfs = []
            
            for label in ['HIGH_RISK', 'MEDIUM_RISK', 'LOW_RISK', 'NO_RISK']:
                label_df = disagreement_df[disagreement_df[f'pred_{label}'] > 0.3]
                if len(label_df) > 0:
                    n_select = min(len(label_df), self.budget_per_week // 4)
                    selected_dfs.append(label_df.head(n_select))
            
            if selected_dfs:
                selected = pd.concat(selected_dfs).drop_duplicates(subset=['idx'])
                selected = selected.head(self.budget_per_week)
            else:
                selected = disagreement_df.head(self.budget_per_week)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Add diversity bonus if requested
        if diversity_bonus and len(selected) < len(disagreement_df):
            selected = self._add_diversity(selected, disagreement_df)
        
        logger.info(f"Selected {len(selected)} samples for review using {strategy} strategy")
        
        return selected
    
    def _add_diversity(
        self,
        selected: pd.DataFrame,
        full_df: pd.DataFrame,
        diversity_factor: float = 0.2
    ) -> pd.DataFrame:
        """Add diversity to selection."""
        n_diverse = int(len(selected) * diversity_factor)
        
        # Get unselected samples
        unselected = full_df[~full_df['idx'].isin(selected['idx'])]
        
        if len(unselected) > 0:
            # Select diverse samples based on different criteria
            diverse_samples = []
            
            # Different text lengths
            if 'text_length' in unselected.columns:
                length_percentiles = [10, 50, 90]
                for pct in length_percentiles:
                    threshold = unselected['text_length'].quantile(pct / 100)
                    samples = unselected[
                        (unselected['text_length'] >= threshold - 50) &
                        (unselected['text_length'] <= threshold + 50)
                    ].head(n_diverse // 3)
                    diverse_samples.append(samples)
            
            # Different slices
            if 'slices' in unselected.columns:
                unique_slices = unselected['slices'].unique()
                for slice_name in unique_slices[:n_diverse // 3]:
                    samples = unselected[unselected['slices'] == slice_name].head(1)
                    diverse_samples.append(samples)
            
            if diverse_samples:
                diverse_df = pd.concat(diverse_samples).drop_duplicates(subset=['idx'])
                diverse_df = diverse_df.head(n_diverse)
                
                # Replace lowest priority samples with diverse ones
                n_keep = len(selected) - len(diverse_df)
                selected = pd.concat([
                    selected.head(n_keep),
                    diverse_df
                ])
        
        return selected
    
    def generate_review_report(
        self,
        selected_df: pd.DataFrame,
        voter_outputs_list: List[Dict[str, Dict]],
        texts: List[str]
    ) -> str:
        """
        Generate human-readable review report.
        
        Args:
            selected_df: Selected samples DataFrame
            voter_outputs_list: Full list of voter outputs
            texts: Full list of texts
            
        Returns:
            Markdown report string
        """
        report_lines = [
            "# Disagreement Review Report",
            f"\n## Summary",
            f"- Total samples selected: {len(selected_df)}",
            f"- Average priority score: {selected_df['priority_score'].mean():.3f}",
            f"- Average entropy: {selected_df['entropy'].mean():.3f}",
            f"- Average abstention rate: {selected_df['abstention_rate'].mean():.3f}",
            "\n## Distribution"
        ]
        
        # Class distribution
        class_dist = []
        for label in ['HIGH_RISK', 'MEDIUM_RISK', 'LOW_RISK', 'NO_RISK']:
            col = f'pred_{label}'
            if col in selected_df.columns:
                avg_pred = selected_df[col].mean()
                class_dist.append(f"- {label}: {avg_pred:.1%}")
        report_lines.extend(class_dist)
        
        # Slice distribution
        if 'slices' in selected_df.columns:
            report_lines.append("\n## Slice Distribution")
            slice_counts = selected_df['slices'].value_counts().head(10)
            for slice_name, count in slice_counts.items():
                report_lines.append(f"- {slice_name}: {count} samples")
        
        # Top disagreements
        report_lines.append("\n## Top Disagreements")
        
        for i, row in selected_df.head(10).iterrows():
            idx = int(row['idx'])
            report_lines.append(f"\n### Sample {idx + 1}")
            report_lines.append(f"**Priority Score:** {row['priority_score']:.3f}")
            report_lines.append(f"**Entropy:** {row['entropy']:.3f}")
            report_lines.append(f"**Text Preview:** {row['text_preview']}")
            
            # Show voter predictions
            voter_outputs = voter_outputs_list[idx]
            report_lines.append("**Voter Predictions:**")
            
            for voter_id, output in voter_outputs.items():
                if output.get('abstain'):
                    report_lines.append(f"- {voter_id}: ABSTAIN")
                else:
                    probs = output.get('probs', {})
                    if probs:
                        pred = max(probs, key=probs.get)
                        conf = probs[pred]
                        report_lines.append(f"- {voter_id}: {pred} ({conf:.2f})")
        
        # Recommendations
        report_lines.append("\n## Recommendations")
        
        # Check for systematic issues
        if selected_df['abstention_rate'].mean() > 0.3:
            report_lines.append("- High abstention rate: Consider adding more training data or adjusting thresholds")
        
        if selected_df['model_disagreement'].mean() > 0.5:
            report_lines.append("- High model disagreement: Different model types are making different predictions")
        
        if 'rare_slice_score' in selected_df.columns and selected_df['rare_slice_score'].mean() > 0.1:
            report_lines.append("- Rare slices present: Consider targeted data collection for underrepresented slices")
        
        return "\n".join(report_lines)
    
    def export_for_labeling(
        self,
        selected_df: pd.DataFrame,
        texts: List[str],
        output_path: str
    ):
        """
        Export selected samples for labeling.
        
        Args:
            selected_df: Selected samples
            texts: Full list of texts
            output_path: Path to save export file
        """
        export_data = []
        
        for _, row in selected_df.iterrows():
            idx = int(row['idx'])
            export_data.append({
                'id': idx,
                'text': texts[idx],
                'priority_score': row['priority_score'],
                'entropy': row['entropy'],
                'predicted_label': None,  # To be filled by human
                'notes': None  # For human annotator notes
            })
        
        # Save as JSON for easy editing
        import json
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(export_data)} samples to {output_path}")