
from typing import List, Optional

from adjustText import adjust_text
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, LinearSegmentedColormap
import numpy as np
import polars as pl
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

import vllm

def plot_semantic_map(doc_target_df: pl.DataFrame, top_num_targets: int = 30) -> plt.Figure:
    """
    Create a semantic map of targets based on stance and frequency.
    This function uses UMAP to reduce the dimensionality of target embeddings
    and visualizes them in a scatter plot with stance represented by color and
    frequency represented by circle size.
    
    Args:
    """
    target_df = doc_target_df.group_by('Target').agg([
        pl.col('Stance').mean().alias('stance_mean'),
        pl.col('Stance').count().alias('count')
    ]).sort('count', descending=True).head(top_num_targets)

    encoder = vllm.LLM('paraphrase-multilingual-MiniLM-L12-v2', task='embed')
    outputs = encoder.embed(target_df['Target'].to_list())
    embeddings = np.stack([o.outputs.embedding for o in outputs], axis=0)
    
    try:
        from cuml.manifold.umap import UMAP
        umap_model = UMAP(spread=0.5)
    except ImportError:
        from umap import UMAP
        umap_model = UMAP()
    coordinates = umap_model.fit_transform(embeddings)
    
    # Create figure with larger size for better visibility
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Define custom green to red color map and normalization
    colors = [(0.7, 0.0, 0.0), (0.9, 0.9, 0.9), (0.0, 0.7, 0.0)]  # green, light gray, red
    cmap = LinearSegmentedColormap.from_list("red_to_green", colors)
    norm = Normalize(vmin=-1, vmax=1)  # Assuming stance ranges from -1 to 1
    
    # Define size scaling based on count
    count_values = target_df['count'].to_numpy()
    size_min, size_max = 100, 500  # Min and max circle sizes
    sizes = size_min + (size_max - size_min) * (count_values - count_values.min()) / (count_values.max() - count_values.min() + 1e-10)
    
    # Create scatter plot
    scatter = ax.scatter(
        coordinates[:, 0], 
        coordinates[:, 1],
        c=target_df['stance_mean'].to_numpy(),
        s=sizes,
        cmap=cmap,
        norm=norm,
        alpha=0.7,
        edgecolors='black'
    )
    
    # Prepare labels for adjustText
    targets = target_df['Target'].to_list()
    texts = []
    for i, (x, y) in enumerate(coordinates):
        # Truncate long target names
        target_label = targets[i]
            
        # Create text objects
        text = ax.text(x, y, target_label, 
                      fontsize=8,
                      ha='center', 
                      bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))
        texts.append(text)
    
    # Use adjustText to prevent label overlap
    adjust_text(texts, 
                force_points=0.2, 
                force_text=0.5,
                expand_points=(1.5, 1.5),
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    
    # Add colorbar for stance
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.02, pad=0.01, aspect=40)
    cbar.set_label('Stance Mean (Against to Favor)')
    
    # Add legend for sizes (3 representative sizes with even numbers)
    # Find min and max counts, and round to even numbers
    min_count = 10 ** (len(str(int(count_values.min()))) - 1)
    max_count = 10 ** (len(str(int(count_values.max()))) - 1)
    
    size_legend_values = [min_count, max_count]
    size_legend_labels = [f'Count: {int(val)}' for val in size_legend_values]
    
    # Calculate the actual sizes that would be used in the plot for these counts
    size_legend_sizes = size_min + (size_max - size_min) * (
        np.array(size_legend_values) - count_values.min()) / (count_values.max() - count_values.min() + 1e-10)
    
    # Create dummy scatter points for legend with correct sizes
    legend_elements = []
    for size, label in zip(size_legend_sizes, size_legend_labels):
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', label=label,
                      markerfacecolor='gray', markersize=np.sqrt(size))
        )
    
    ax.legend(handles=legend_elements, title="Target Frequency", loc="upper right")
    
    # Set labels and title
    # ax.set_title('Semantic Map of Targets with Stance and Frequency', fontsize=16)
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    
    # Remove ticks as they don't have meaningful values
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Save figure
    plt.tight_layout()

    return fig



def _sort_by_similarity_bidirectional(items, similarity_func, start_item=None):
    """
    Alternative approach that builds the chain from both ends.
    
    This can sometimes produce better results by allowing items to be added
    to either end of the current chain.
    """
    if not items:
        return []
    
    if len(items) == 1:
        return items[:]
    
    remaining = items[:]
    
    # Choose starting item
    if start_item is not None and start_item in remaining:
        current = start_item
        remaining.remove(current)
    else:
        current = remaining.pop(0)
    
    chain = [current]
    
    while remaining:
        best_similarity = float('-inf')
        best_item = None
        best_idx = -1
        add_to_end = True
        
        # Check similarity to both ends of the chain
        for i, item in enumerate(remaining):
            # Similarity to start of chain
            sim_start = similarity_func(chain[0], item)
            if sim_start > best_similarity:
                best_similarity = sim_start
                best_item = item
                best_idx = i
                add_to_end = False
            
            # Similarity to end of chain
            sim_end = similarity_func(chain[-1], item)
            if sim_end > best_similarity:
                best_similarity = sim_end
                best_item = item
                best_idx = i
                add_to_end = True
        
        item = remaining.pop(best_idx)
        if add_to_end:
            chain.append(item)
        else:
            chain.insert(0, item)
    
    return chain


def _identify_significant_transitions(
        df: pl.DataFrame, 
        filter_col: str,
        min_flow_count=15, 
        min_posts_per_user=10
    ):
    """
    Identify significant user transitions using changepoint detection.
    
    Parameters:
    -----------
    df : polars.DataFrame
        DataFrame with columns: <filter_col>, createtime, Target, Stance
    min_flow_count : int
        Minimum number of users making similar transitions to consider significant
    
    Returns:
    --------
    polars.DataFrame
        DataFrame with significant transitions detected via changepoint analysis
    """
    
    # Process each user individually
    transitions_df = pl.DataFrame()
    
    for user_df in tqdm(df.partition_by(filter_col), desc="Processing users transitions"):
        user_df = user_df.sort('createtime')
        
        if len(user_df) < min_posts_per_user:
            continue
            
        user_transitions = user_df.group_by_dynamic('createtime', every='1mo', period='2mo')\
            .agg([pl.col(filter_col).first(), pl.col('Target').mode()])\
            .with_columns(pl.col('Target').list.get(0))\
            .with_columns(pl.col('Target').shift(1).alias('from_target'))\
            .rename({'Target': 'to_target'})\
            .filter(pl.col('to_target') != pl.col('from_target'))
        transitions_df = pl.concat([transitions_df, user_transitions], how='diagonal_relaxed')

    if transitions_df.is_empty():
        print("No changepoints detected across users")
        return pl.DataFrame()
    
    # Convert to DataFrame and aggregate significant patterns
    print(f"Found {len(transitions_df)} individual changepoint transitions")
    
    # Aggregate transitions to find significant patterns
    significant_transitions = transitions_df.group_by(['createtime', 'to_target', 'from_target'])\
        .agg(pl.col(filter_col).count().alias('transition_count'))\
        .filter(pl.col('transition_count') >= min_flow_count)
    
    print(f"Identified {len(significant_transitions)} significant transition patterns")
    return significant_transitions
    

class StanceTrendDiagram:
    def __init__(self, figsize=(18, 12)):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.trend_data = {}  # Cache for loaded trend data
        self.sentence_model = None
        
        # Enhanced color scheme for better trend visualization
        self.stance_colors = {
            'positive': '#2E8B57',  # Sea green
            'negative': '#DC143C',  # Crimson
            'neutral': '#708090'    # Slate gray
        }
        
        # Neutral color for transitions
        self.transition_color = (0.6, 0.6, 0.6, 0.5)  # Light gray with transparency

    def load_data(self, df: pl.DataFrame, trend_df: pl.DataFrame, filter_col: str):
        """
        Load and process stance data. 
        Expected columns: <filter_col>, createtime, Target, Stance
        """
        self.df = df

        targets = self.df['Target'].unique().to_list()
        trend_data = {}
        
        for target in targets:
            trend_data[target] = trend_df.filter(pl.col('Target') == target).sort('createtime')
        
        self.trend_data = trend_data
            
        # Ensure we have the required columns
        required_cols = [filter_col, 'createtime', 'Target', 'Stance']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        self.min_time = self.df['createtime'].min()
        self.max_time = self.df['createtime'].max()
            
        return self.df
    
    def time_to_normalized(self, time):
        """
        Convert a datetime to a normalized value between 0 and 1 based on the dataset's time range.
        """
        if self.min_time is None or self.max_time is None:
            raise ValueError("Time range not set. Load data first.")
        
        return (time - self.min_time) / (self.max_time - self.min_time)
    
    def get_target_ordering(self, targets: List[str], significant_transitions: pl.DataFrame):
        """
        Order targets semantically using sentence embeddings + UMAP
        """
        target_transitions = significant_transitions.group_by(['to_target', 'from_target'])\
            .agg(pl.col('transition_count').sum())\
            .rows_by_key(key=['from_target', 'to_target'], unique=True)

        target_transitions = {frozenset(k): v[0] for k, v in target_transitions.items()}
            
        if self.sentence_model is None:
            print("Loading sentence transformer model...")
            self.sentence_model = vllm.LLM('all-MiniLM-L6-v2', task='embed')
        
        # Get embeddings for target names
        embeddings = self.sentence_model.encode(targets)
        target_embeddings = {target: embedding for target, embedding in zip(targets, embeddings)}
        
        def get_similarity(a, b):
            cos_sim = np.dot(target_embeddings[a], target_embeddings[b]) / (np.linalg.norm(target_embeddings[a]) * np.linalg.norm(target_embeddings[b]))
            transition_sim = target_transitions.get(frozenset([a, b]), 0)
            return transition_sim + cos_sim

        ordered_targets = _sort_by_similarity_bidirectional(
            targets, 
            get_similarity
        )
        return ordered_targets
    
    def prepare_stream_data(self, n_points=200):
        """
        Prepare continuous stream data for each target
        """
        stream_data = {}
        
        for target, trend_df in self.trend_data.items():
            if len(trend_df) == 0:
                continue
                
            # Get time points and normalize
            trend_df = trend_df.with_columns(pl.col('createtime').cast(pl.Datetime).dt.replace_time_zone('UTC'))
            trend_times = trend_df['createtime'].dt.timestamp().to_numpy()
            trend_times_norm = (trend_times - trend_times.min()) / (trend_times.max() - trend_times.min())
            
            time_range = np.linspace(
                self.time_to_normalized(trend_df['createtime'].min()),
                self.time_to_normalized(trend_df['createtime'].max()),
                n_points
        )

            # Get volume and trend data
            volumes = trend_df['volume'].to_numpy()
            trends = trend_df['trend_mean'].to_numpy()
            
            # Interpolate to common time grid
            if len(trend_times_norm) > 1:
                volume_interp = interp1d(trend_times_norm, volumes, 
                                       kind='cubic', bounds_error=False, fill_value=0)
                trend_interp = interp1d(trend_times_norm, trends, 
                                      kind='cubic', bounds_error=False, fill_value=0)
                
                # Smooth the interpolated data
                interpolated_volumes = gaussian_filter1d(volume_interp(time_range), sigma=2)
                interpolated_trends = gaussian_filter1d(trend_interp(time_range), sigma=2)
            else:
                interpolated_volumes = np.full(n_points, volumes[0])
                interpolated_trends = np.full(n_points, trends[0])
            
            # Ensure non-negative volumes
            interpolated_volumes = np.maximum(interpolated_volumes, 0)
            
            stream_data[target] = {
                'time': time_range,
                'volume': interpolated_volumes,
                'trend': interpolated_trends,
                'volume_normalized': interpolated_volumes / np.max(interpolated_volumes) if np.max(interpolated_volumes) > 0 else interpolated_volumes
            }
        
        return stream_data
    
    def _calculate_distribution_change(self, dist1, dist2):
        """Calculate the magnitude of change between two target distributions using Total Variation Distance"""
        all_targets = set(dist1.keys()) | set(dist2.keys())
        
        total_variation = 0
        for target in all_targets:
            prob1 = dist1.get(target, 0)
            prob2 = dist2.get(target, 0)
            total_variation += abs(prob2 - prob1)
        
        return total_variation / 2  # Normalize to [0, 1]
    
    def _find_main_focus_shift(self, dist1, dist2):
        """Identify the primary focus shift between two distributions"""
        all_targets = set(dist1.keys()) | set(dist2.keys())
        
        # Calculate changes for each target
        changes = {}
        for target in all_targets:
            prob1 = dist1.get(target, 0)
            prob2 = dist2.get(target, 0)
            changes[target] = prob2 - prob1
        
        # Find the largest decrease (what they moved away from)
        decreases = {t: -change for t, change in changes.items() if change < -0.1}
        # Find the largest increase (what they moved toward)
        increases = {t: change for t, change in changes.items() if change > 0.1}
        
        if decreases and increases:
            from_target = max(decreases, key=decreases.get)
            to_target = max(increases, key=increases.get)
            
            # Calculate the magnitude of this specific shift
            shift_magnitude = min(decreases[from_target], increases[to_target])
            
            return from_target, to_target, shift_magnitude
        
        return None, None, 0
    
    def _trend_to_color_gradient(self, trend_values, alpha=0.8):
        """Convert trend values to color gradient for streams"""
        colors = []
        green = np.array([0.0, 1.0, 0.0])
        red = np.array([1.0, 0.0, 0.0])
        grey = np.array([0.5, 0.5, 0.5])
        for trend in trend_values:
            if trend >= 0.0:
                # Positive trend - shades of green
                intensity = min(abs(trend), 1.0)
                colour = intensity * green + (1 - intensity) * grey
                colors.append((*colour, alpha))
            else:
                # Negative trend - shades of red
                intensity = min(abs(trend), 1.0)
                colour = intensity * red + (1 - intensity) * grey
                colors.append((*colour, alpha))
        return colors
    
    def _get_targets(self):
        targets = self.df['Target'].unique().to_list()
        targets = [t for t in targets if t in self.trend_data]
        assert len(targets) > 0, "No valid targets found in the data."
        return targets

    def plot_trends(self, max_stream_width=3.0, flow_alpha=0.6, 
                               min_transition_count=15):
        """Create the enhanced continuous stream diagram with integrated flows"""
        targets = self._get_targets()
        
        # Prepare stream data
        stream_data = self.prepare_stream_data()
        significant_transitions = _identify_significant_transitions(
            self.df,
            min_flow_count=min_transition_count,
            min_posts_per_user=10
        )

        targets = self.get_target_ordering(targets, significant_transitions)
        
        # Set up the plot
        self.ax.clear()
        n_targets = len(targets)
        target_spacing = 1.2  # Slightly increased spacing for better flow visibility
        
        # Calculate vertical positions for targets
        target_positions = {}
        current_y = 0
        
        # Plot continuous streams for each target with gradient coloring
        for i, target in enumerate(targets):
            if target not in stream_data:
                continue
                
            data = stream_data[target]
            time_points = data['time']
            volumes = data['volume_normalized'] * max_stream_width
            trends = data['trend']
            
            # Create stream boundaries
            upper_boundary = current_y + volumes / 2
            lower_boundary = current_y - volumes / 2
            
            # Store position for flows
            target_positions[target] = current_y
            
            # Create gradient-colored stream segments
            n_segments = len(time_points) - 1
            for j in range(n_segments):
                # Create polygon for this segment
                x_segment = [time_points[j], time_points[j+1], time_points[j+1], time_points[j]]
                y_segment = [upper_boundary[j], upper_boundary[j+1], lower_boundary[j+1], lower_boundary[j]]
                
                # Get color based on trend
                trend_color = self._trend_to_color_gradient([trends[j]])[0]
                
                # Draw segment
                segment_polygon = mpatches.Polygon(list(zip(x_segment, y_segment)), 
                                        facecolor=trend_color, 
                                        edgecolor=trend_color, linewidth=0.2, zorder=2)
                self.ax.add_patch(segment_polygon)
            
            # Add target label with better positioning
            label_y = current_y
            bbox_props = dict(boxstyle="round,pad=0.4", facecolor='white', 
                            alpha=0.95, edgecolor='gray', linewidth=1)
            self.ax.text(-0.08, label_y, target, va='center', ha='right', 
                        fontweight='bold', fontsize=11, bbox=bbox_props, zorder=5)
            
            # Update y position for next target
            current_y += target_spacing + max_stream_width
        
        # Plot significant transitions as smooth integrated flows
        if len(significant_transitions) > 0:
            print(f"Drawing {len(significant_transitions)} significant transitions...")
            
            for row in significant_transitions.iter_rows(named=True):
                from_target = row['from_target']
                to_target = row['to_target']
                
                if from_target in target_positions and to_target in target_positions:
                    from_y = target_positions[from_target]
                    to_y = target_positions[to_target]
                    flow_time = self.time_to_normalized(row['createtime'])
                    flow_count = row['transition_count']
                    
                    # Draw smooth transition flow
                    self._draw_integrated_transition_flow(
                        flow_time, from_y, to_y, 
                        flow_count, alpha=flow_alpha
                    )
        
        # Customize plot
        self.ax.set_xlim(-0.12, 1.05)
        self.ax.set_ylim(-max_stream_width, current_y + max_stream_width)
        
        self.ax.set_xlabel('Time →', fontsize=14, fontweight='bold')
        self.ax.set_ylabel('Stance Targets', fontsize=14, fontweight='bold')
        self.ax.set_title('Enhanced Stance Streams: Trend-Colored Activity with Significant User Transitions\n' +
                         '(Stream width = Volume, Color gradient = Stance trend, Gray flows = User movement)', 
                         fontsize=15, fontweight='bold', pad=25)
        
        # Remove ticks
        n_x_ticks = 10
        labels = [d.strftime('%Y-%m') for d in np.linspace(self.min_time, self.max_time, n_x_ticks)]
        self.ax.set_xticks(np.linspace(0, 1, n_x_ticks), labels=labels)
        self.ax.set_yticks([])
        
        # Add enhanced legend
        self._add_enhanced_legend()
        
        # Remove spines
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        
        plt.tight_layout()
        return self.fig
    
    def _draw_integrated_transition_flow(self, time_x, from_y, to_y, flow_count, 
                                       max_flow_width=0.8, alpha=0.6):
        """Draw smooth transition flows that integrate well with streams"""
        if flow_count < 5:
            return
            
        # Calculate flow width with better scaling
        flow_width = min(np.sqrt(flow_count) * 0.08, max_flow_width)
        
        if flow_width < 0.03:  # Skip very small flows
            return
        
        # Create extended smooth curve that blends with streams
        flow_length = 0.08  # Longer flow for better integration
        n_points = 100
        curve_x = np.linspace(time_x - flow_length/2, time_x + flow_length/2, n_points)
        
        # Smoother bezier-like curve
        t = np.linspace(0, 1, n_points)
        # Control points for smooth S-curve
        control_y = (from_y + to_y) / 2
        curve_y = ((1-t)**3 * from_y + 
                  3*(1-t)**2*t * (from_y + (control_y - from_y) * 0.3) + 
                  3*(1-t)*t**2 * (to_y + (control_y - to_y) * 0.3) + 
                  t**3 * to_y)
        
        # Create tapered flow (wider in middle, narrower at ends)
        taper = 4 * t * (1 - t)  # Bell curve shape
        half_width = flow_width * taper / 2
        
        upper_y = curve_y + half_width
        lower_y = curve_y - half_width
        
        # Create smooth gradient for the flow
        vertices = np.column_stack([
            np.concatenate([curve_x, curve_x[::-1]]),
            np.concatenate([upper_y, lower_y[::-1]])
        ])
        
        # Use neutral color for transitions
        flow_polygon = mpatches.Polygon(vertices, facecolor=self.transition_color, 
                             edgecolor='none', zorder=3, alpha=alpha)
        self.ax.add_patch(flow_polygon)
    
    def _add_enhanced_legend(self):
        """Add enhanced legend for the visualization"""
        legend_elements = [
            mpatches.Patch(color=(0.1, 0.7, 0.1), label='Favor Stance Trend'),
            mpatches.Patch(color=(0.5, 0.5, 0.5), label='Neutral Stance Trend'),
            mpatches.Patch(color=(0.7, 0.1, 0.1), label='Against Stance Trend'),
            mpatches.Patch(color=self.transition_color[:3], alpha=0.7, label='Significant User Transitions'),
        ]
        
        legend = self.ax.legend(handles=legend_elements, loc='upper right', 
                               bbox_to_anchor=(1.18, 1), frameon=True, 
                               fancybox=True, shadow=True)
        legend.get_frame().set_alpha(0.9)

def plot_trend_map(
        document_df: pl.DataFrame,
        trend_df: pl.DataFrame,
        figsize=(20, 14),
        plot_stream_transitions=True,
        filter_col: Optional[str] = None,
        max_stream_width=4.0,
        min_transition_count=4
    ):
    """
    Plot the continuous stance stream diagram with enhanced features.

    Args:
        document_df (pl.DataFrame): DataFrame containing document stance data with columns:

    """
    viz = StanceTrendDiagram(figsize=figsize)
    viz.load_data(document_df, trend_df)

    fig = viz.plot_trends(
        max_stream_width=max_stream_width,
        min_transition_count=min_transition_count
    )
    
    return fig