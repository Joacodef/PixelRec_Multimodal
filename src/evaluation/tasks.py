from enum import Enum
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed


class EvaluationTask(Enum):
    """
    Defines the available evaluation tasks for recommender systems.

    This Enum provides a standardized way to specify the type of evaluation
    to be performed, enabling clear distinction between different evaluation
    methodologies.
    """
    TOP_K_RETRIEVAL = "top_k_retrieval"
    TOP_K_RANKING = "top_k_ranking"


class BaseEvaluator(ABC):
    """
    Base class for all recommender system evaluators.

    This abstract class defines the common interface and shared properties
    for all specific evaluation tasks. It ensures that all evaluators have
    access to the recommender model, test data, and configuration, and
    provide a standardized method for reporting results.
    """
    
    def __init__(self, recommender, test_data: pd.DataFrame, config, task_name: str, **kwargs):
        """
        Initializes the BaseEvaluator.

        Args:
            recommender: The recommender system object to be evaluated. This object
                         is expected to have methods like `get_recommendations` or
                         `get_item_score`.
            test_data: A pandas DataFrame containing the test set interactions.
                       It must include 'user_id' and 'item_id' columns.
            config: The configuration object, typically an instance of `Config`,
                    containing global parameters like `top_k`.
            task_name: A string identifying the specific evaluation task (e.g., "Top-K Retrieval").
            **kwargs: Additional keyword arguments that might be specific to a subclass
                      (e.g., `filter_seen`, `num_workers`).
        """
        self.recommender = recommender
        self.test_data = test_data
        self.config = config
        self.task_name = task_name
        # Retrieves the `top_k` value from the configuration, defaulting to 50 if not specified.
        self.top_k = getattr(config.recommendation, 'top_k', 50)
        # Determines whether previously seen items should be filtered from recommendations.
        self.filter_seen = kwargs.get('filter_seen', True)
        
        # Ensures that 'user_id' and 'item_id' columns in the test data are treated as strings
        # for consistent handling across the system.
        self.test_data = self.test_data.copy()
        self.test_data['user_id'] = self.test_data['user_id'].astype(str)
        self.test_data['item_id'] = self.test_data['item_id'].astype(str)
        
    @abstractmethod
    def evaluate(self) -> Dict[str, Any]:
        """
        Abstract method to perform the evaluation specific to the task.

        This method must be implemented by all concrete evaluator subclasses.
        It orchestrates the process of generating recommendations or scores
        and computing the relevant metrics.

        Returns:
            A dictionary containing the calculated evaluation metrics and
            potentially other results (e.g., raw predictions).
        """
        pass
    
    def print_summary(self, results: Dict[str, Any]):
        """
        Prints a formatted summary of the evaluation results.

        This method iterates through the calculated metrics and displays them
        in a readable format to the console.

        Args:
            results: A dictionary containing the evaluation metrics, typically
                     the output of the `evaluate` method.
        """
        print(f"\n=== {self.task_name} Results ===")
        # Iterates through all key-value pairs in the results dictionary.
        for metric, value in results.items():
            # Excludes metadata and raw predictions from the main summary printout.
            if metric not in ['evaluation_metadata', 'predictions']:
                # Formats float values to four decimal places for readability.
                if isinstance(value, float):
                    print(f"{metric}: {value:.4f}")
                else:
                    print(f"{metric}: {value}")


class TopKRetrievalEvaluator(BaseEvaluator):
    """
    Evaluates Top-K retrieval performance of a recommender system.

    This evaluator measures how effectively the model retrieves relevant items
    from a set of candidates (which may include negative samples). It computes
    metrics such as Precision@k, Recall@k, F1-score, Hit Rate, NDCG, and MRR.
    It supports parallel processing for efficiency and various negative sampling strategies.
    """
    
    def __init__(self, recommender, test_data: pd.DataFrame, config, 
                 use_sampling: bool = True, num_negatives: int = 100, 
                 sampling_strategy: str = 'random', **kwargs):
        """
        Initializes the TopKRetrievalEvaluator.

        Args:
            recommender: The recommender system object.
            test_data: A pandas DataFrame containing the test set interactions.
            config: The configuration object.
            use_sampling: If True, negative samples are generated for each user
                          to create a candidate set. If False, all unseen items
                          are considered candidates (full evaluation).
            num_negatives: The number of negative items to sample per positive item
                           when `use_sampling` is True.
            sampling_strategy: The strategy for sampling negative items ('random',
                               'popularity', 'popularity_inverse').
            **kwargs: Additional keyword arguments, including `num_workers` for
                      parallel processing.
        """
        # Initializes the base evaluator with the task name.
        super().__init__(recommender, test_data, config, "Top-K Retrieval", **kwargs)
        self.use_sampling = use_sampling
        self.num_negatives = num_negatives
        self.sampling_strategy = sampling_strategy
        # Number of parallel workers for processing users, defaults to 1 (sequential).
        self.num_workers = kwargs.get('num_workers', 1)
        
    def _get_all_item_ids(self) -> List[str]:
        """
        Retrieves all available item IDs from the recommender's dataset or test data.

        This method attempts to get the full catalog of item IDs from the recommender's
        internal dataset (e.g., via a LabelEncoder) for the most comprehensive list.
        As a fallback, it extracts unique item IDs directly from the test data.

        Returns:
            A list of all unique item IDs (string) in the system.
        """
        # Attempts to retrieve item IDs from the recommender's dataset's item encoder.
        if (hasattr(self.recommender, 'dataset') and 
            hasattr(self.recommender.dataset, 'item_encoder') and
            hasattr(self.recommender.dataset.item_encoder, 'classes_') and
            self.recommender.dataset.item_encoder.classes_ is not None):
            return [str(item_id) for item_id in self.recommender.dataset.item_encoder.classes_]
        
        # Fallback: Returns unique item IDs present in the test data if a more comprehensive
        # source is not available.
        return list(self.test_data['item_id'].unique())
        
    def _sample_negatives(self, user_id: str, positive_items: List[str]) -> List[str]:
        """
        Samples a specified number of negative (non-interacted) items for a given user.

        This method ensures that sampled negative items are not among the user's
        known positive interactions and can employ different sampling strategies
        to select items (random, popularity-biased, or inverse popularity-biased).

        Args:
            user_id: The ID (string) of the user for whom to sample negatives.
            positive_items: A list of item IDs (string) that the user has positively
                            interacted with in the test set. These will be excluded
                            from the negative sample pool.

        Returns:
            A list of sampled negative item IDs (string). The number of items
            is determined by `self.num_negatives`.
        """
        # Retrieves all available item IDs in the system.
        all_items = self._get_all_item_ids()
        # Converts positive items to a set for efficient lookup.
        positive_items_set = set(str(item) for item in positive_items)
        
        # Filters out positive items from the global item list to create negative candidates.
        negative_candidates = [item for item in all_items if item not in positive_items_set]
        
        # If the number of available negative candidates is less than required, returns all of them.
        if len(negative_candidates) < self.num_negatives:
            return negative_candidates
        
        # Returns an empty list if no negative candidates are available.
        if not negative_candidates:
            return []
        
        # Sets a deterministic seed for random operations based on the user ID.
        # This ensures reproducibility of sampled negatives for the same user across runs.
        user_seed = hash(str(user_id)) % (2**31)  # Ensures a positive 32-bit integer for consistency.
        np.random.seed(user_seed)
        random.seed(user_seed)

        if self.sampling_strategy == 'random':
            # Randomly samples the required number of negative items without replacement.
            num_to_sample = min(self.num_negatives, len(negative_candidates))
            return random.sample(negative_candidates, num_to_sample)
            
        elif self.sampling_strategy == 'popularity':
            # Samples negative items biased by their popularity (more popular items are more likely).
            item_counts = self.test_data['item_id'].astype(str).value_counts()
            
            weights = []
            valid_candidates = []
            
            # Calculates weights for each negative candidate based on its popularity.
            for item in negative_candidates:
                item_str = str(item)
                # Defaults to a count of 1 for items not found in `item_counts` to avoid zero weights.
                count = item_counts.get(item_str, 1)  
                if count > 0:  
                    weights.append(float(count))
                    valid_candidates.append(item_str)
            
            # Returns an empty list if no valid candidates exist after filtering.
            if not valid_candidates:
                return []
            
            # Normalizes weights to form a probability distribution.
            weights = np.array(weights, dtype=np.float64)
            if weights.sum() == 0:
                # Fallback to uniform sampling if all weights are zero.
                weights = np.ones(len(weights))
            weights = weights / weights.sum()
            
            # Samples from valid candidates based on the calculated probabilities.
            num_to_sample = min(self.num_negatives, len(valid_candidates))
            try:
                sampled_indices = np.random.choice(
                    len(valid_candidates), 
                    size=num_to_sample, 
                    replace=False, 
                    p=weights
                )
                return [valid_candidates[i] for i in sampled_indices]
            except ValueError as e:
                # If sampling fails due to distribution issues, falls back to random sampling.
                print(f"Warning: Popularity sampling failed for user {user_id}: {e}. Using random sampling.")
                return random.sample(valid_candidates, num_to_sample)
                
        else:  # 'popularity_inverse'
            # Samples negative items biased by inverse popularity (less popular items are more likely).
            item_counts = self.test_data['item_id'].astype(str).value_counts()
            
            weights = []
            valid_candidates = []
            
            # Calculates inverse weights for each negative candidate.
            for item in negative_candidates:
                item_str = str(item)
                count = item_counts.get(item_str, 1)  # Default count of 1 for unseen items.
                if count > 0:  
                    # Inverse weight: less popular items get higher weight.
                    inv_weight = 1.0 / float(count)
                    weights.append(inv_weight)
                    valid_candidates.append(item_str)
            
            if not valid_candidates:
                return []
            
            # Normalizes weights to probabilities.
            weights = np.array(weights, dtype=np.float64)
            if weights.sum() == 0:
                # Fallback to uniform sampling if all weights are zero.
                weights = np.ones(len(weights))
            weights = weights / weights.sum()
            
            # Samples from valid candidates based on inverse probabilities.
            num_to_sample = min(self.num_negatives, len(valid_candidates))
            try:
                sampled_indices = np.random.choice(
                    len(valid_candidates), 
                    size=num_to_sample, 
                    replace=False, 
                    p=weights
                )
                return [valid_candidates[i] for i in sampled_indices]
            except ValueError as e:
                # If sampling fails, falls back to random sampling.
                print(f"Warning: Inverse popularity sampling failed for user {user_id}: {e}. Using random sampling.")
                return random.sample(valid_candidates, num_to_sample)

    def _process_user(self, user_id_and_interactions: Tuple[str, pd.DataFrame]) -> Tuple[str, List[Tuple[str, float]], List[str], List[str]]:
        """
        Processes a single user to obtain recommendations and associated data.

        This corrected method ensures the single ground-truth positive item from the
        test set is always included in the candidate list for ranking, and that the
        recommender is instructed not to filter it out. This is essential for
        obtaining meaningful metrics in a leave-one-out evaluation.
        """
        user_id, user_interactions = user_id_and_interactions
        user_id = str(user_id)

        # 1. IDENTIFY GROUND TRUTH POSITIVE ITEMS
        # In leave-one-out, this will be a list with a single item.
        positive_items = [str(item) for item in user_interactions['item_id'].tolist()]
        if not positive_items:
            # If for some reason there's no positive item, return empty.
            return user_id, [], [], []

        # 2. CONSTRUCT THE CANDIDATE SET
        # Start with the ground truth item(s).
        candidate_items = list(positive_items)

        if self.use_sampling:
            # Generate negative samples, ensuring they don't include the positive item.
            negative_items = self._sample_negatives(user_id, positive_items)
            candidate_items.extend(negative_items)
        
        # De-duplicate and shuffle the final list to avoid any order bias.
        candidate_items = list(dict.fromkeys(candidate_items))
        user_shuffle_seed = hash(str(user_id) + "shuffle") % (2**31)
        local_random = random.Random(user_shuffle_seed)
        local_random.shuffle(candidate_items)

        try:
            # 3. GET RECOMMENDATIONS WITH `filter_seen=False`
            # This is the most critical change. We pass our perfectly crafted
            # candidate set (1 positive + N negatives) and explicitly tell the
            # recommender NOT to filter any items from it.
            recommendations = self.recommender.get_recommendations(
                user_id=user_id,
                top_k=self.top_k,
                filter_seen=False, # CRITICAL: Prevents filtering the positive item.
                candidates=candidate_items
            )
            
            if recommendations:
                recommendations = [(str(item_id), score) for item_id, score in recommendations]
            
            recommended_items_only = [item_id for item_id, _ in recommendations]
            return user_id, recommendations, positive_items, recommended_items_only

        except Exception as e:
            print(f"Error evaluating user {user_id}: {e}")
            return user_id, [], positive_items, []
    # --- END OF CORRECTED METHOD ---


    def evaluate(self) -> Dict[str, Any]:
        """
        Executes the Top-K retrieval evaluation process.

        This method groups test data by user, processes each user's recommendations
        (potentially in parallel), and then calculates vectorized metrics across all users.

        Returns:
            A dictionary containing the calculated evaluation metrics (e.g., avg_precision_at_k,
            avg_recall_at_k, avg_ndcg_at_k, avg_mrr) and raw predictions.
        """
        print(f"Evaluating Top-K Retrieval (K={self.top_k})")
        if self.use_sampling:
            print(f"Using negative sampling: {self.num_negatives} negatives per user, strategy: {self.sampling_strategy}")

        # Groups test data by user to process each user individually.
        user_groups = list(self.test_data.groupby('user_id'))
        num_users = len(user_groups)

        # Sets global seeds for NumPy and Python's random module to ensure reproducible evaluation order.
        np.random.seed(42)
        random.seed(42)

        raw_results = []
        # Checks if parallel processing is enabled and beneficial (more than one worker and user).
        if self.num_workers > 1 and num_users > 1:
            # Attempts to use the 'fork' context for multiprocessing, which is efficient on Unix-like systems.
            try:
                mp_context = mp.get_context('fork')
                print(f"Using 'fork' context for parallel processing with {self.num_workers} workers.")
            except (ValueError, RuntimeError):
                # Falls back to 'spawn' if 'fork' is not available or causes issues (common on Windows).
                mp_context = mp.get_context('spawn')
                print(f"Warning: 'fork' context not available. Falling back to 'spawn'. Parallel evaluation may be slow.")

            # Uses a process pool executor for parallel execution of `_process_user`.
            with ProcessPoolExecutor(max_workers=self.num_workers, mp_context=mp_context) as executor:
                futures = [executor.submit(self._process_user, user_group) for user_group in user_groups]
                # Displays a progress bar for parallel user evaluation.
                for future in tqdm(as_completed(futures), total=num_users, desc="Evaluating users (parallel)"):
                    raw_results.append(future.result())
        else:
            print("Evaluating users (sequential)...")
            # Performs sequential processing if parallel workers are not configured or not beneficial.
            raw_results = [self._process_user(user_group) for user_group in tqdm(user_groups)]

        # --- Vectorized Metric Calculation ---
        # Unpacks the raw results collected from user processing.
        user_ids = [res[0] for res in raw_results]
        all_predictions = {res[0]: res[1] for res in raw_results} # Stores raw recommendations for each user.
        all_pos_items = [res[2] for res in raw_results] # True positive items for each user.
        all_rec_items = [res[3] for res in raw_results] # Recommended items for each user.
        
        # Initializes NumPy arrays to store metric components for vectorized calculations.
        hits_at_k = np.zeros(num_users)
        # Denominators for precision, representing the number of recommendations made for each user.
        precision_denominators = np.array([len(r) for r in all_rec_items], dtype=np.float32)
        # Denominators for recall, representing the total number of relevant items for each user.
        recall_denominators = np.array([len(p) for p in all_pos_items], dtype=np.float32)
        mrr = np.zeros(num_users)
        ndcg_at_k = np.zeros(num_users)

        # Iterates through each user's results to compute per-user hits, MRR, and NDCG.
        for i in range(num_users):
            rec_set = set(all_rec_items[i]) # Converts recommended items to a set for efficient intersection.
            pos_set = set(all_pos_items[i]) # Converts positive items to a set.
            
            if not pos_set:
                # Skips metric calculation if there are no positive items for the user.
                continue

            # Calculates the number of hits (relevant items in recommendations).
            num_hits = len(rec_set.intersection(pos_set))
            hits_at_k[i] = num_hits
            
            # Calculates Mean Reciprocal Rank (MRR). Finds the rank of the first relevant item.
            for j, item in enumerate(all_rec_items[i], 1):
                if item in pos_set:
                    mrr[i] = 1.0 / j
                    break # Stops after the first hit for MRR.
            
            # Calculates Normalized Discounted Cumulative Gain (NDCG).
            ndcg_at_k[i] = self._calculate_ndcg(all_rec_items[i], pos_set, self.top_k)
        
        # Performs vectorized calculations for Precision and Recall.
        with np.errstate(divide='ignore', invalid='ignore'): # Suppresses division by zero warnings.
            precision = hits_at_k / precision_denominators
            recall = hits_at_k / recall_denominators
        
        # Sets NaN values (resulting from division by zero, e.g., no recommendations made) to 0.0.
        precision[np.isnan(precision)] = 0.0
        recall[np.isnan(recall)] = 0.0

        # Calculates F1-score, handling potential division by zero for (precision + recall).
        with np.errstate(divide='ignore', invalid='ignore'):
            f1 = 2 * precision * recall / (precision + recall)
        f1[np.isnan(f1)] = 0.0
        
        # Calculates Hit Rate (percentage of users with at least one hit).
        hit_rate = (hits_at_k > 0).astype(float)
        
        # Aggregates all calculated metrics into a single dictionary.
        results = {
            "avg_precision_at_k": np.mean(precision),
            "avg_recall_at_k": np.mean(recall),
            "avg_f1_at_k": np.mean(f1),
            "avg_hit_rate_at_k": np.mean(hit_rate),
            "avg_ndcg_at_k": np.mean(ndcg_at_k),
            "avg_mrr": np.mean(mrr)
        }
        
        # Adds metadata about the evaluation run to the results.
        results['num_users_evaluated'] = num_users
        results['evaluation_method'] = 'negative_sampling' if self.use_sampling else 'full_evaluation'
        results['predictions'] = all_predictions # Includes raw predictions for further analysis.
        
        return results
    
    def _calculate_ndcg(self, recommended_items: List[str], relevant_items: set, k: int) -> float:
        """
        Calculates Normalized Discounted Cumulative Gain at K (NDCG@k) for a single user.

        Args:
            recommended_items: A list of item IDs (string) recommended to the user, in ranked order.
            relevant_items: A set of item IDs (string) that are truly relevant to the user.
            k: The number of top recommendations to consider.

        Returns:
            The NDCG@k score for the user (float). Returns 0.0 if no relevant items are provided.
        """
        if not relevant_items:
            return 0.0
        
        # Calculates Discounted Cumulative Gain (DCG).
        dcg = 0.0
        # Iterates through the top 'k' recommended items with 1-based indexing for log2.
        for i, item in enumerate(recommended_items[:k], 1):
            if item in relevant_items:
                # Adds relevance score (1.0 for binary relevance) discounted by position.
                dcg += 1.0 / np.log2(i + 1)
        
        # Calculates Ideal Discounted Cumulative Gain (IDCG) for a perfect ranking.
        # It assumes all relevant items are ranked perfectly at the top.
        num_relevant = min(len(relevant_items), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(num_relevant))
        
        # Returns NDCG (DCG / IDCG). Handles division by zero if IDCG is 0.
        return dcg / idcg if idcg > 0 else 0.0


class TopKRankingEvaluator(BaseEvaluator):
    """
    Evaluates Top-K ranking performance of a recommender system.

    This evaluator focuses on how well the model ranks a given set of relevant
    test items for a user. It assumes all test items are relevant and measures
    metrics like average rank, median rank, MRR, Hit Rate, and NDCG based on
    the model's predicted scores for these items.
    """
    
    def __init__(self, recommender, test_data: pd.DataFrame, config, **kwargs):
        """
        Initializes the TopKRankingEvaluator.

        Args:
            recommender: The recommender system object. This object is expected
                         to have a `get_item_score` method that returns a score
                         for a given user-item pair.
            test_data: A pandas DataFrame containing the test set interactions.
                       For ranking tasks, all items in `test_data` for a user
                       are considered relevant.
            config: The configuration object.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(recommender, test_data, config, "Top-K Ranking", **kwargs)
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Executes the Top-K ranking evaluation process.

        This method iterates through each user in the test data, obtains scores
        for their associated test items from the recommender, ranks these items,
        and computes various ranking metrics.

        Returns:
            A dictionary containing the calculated evaluation metrics (e.g., avg_avg_rank,
            avg_median_rank, avg_ndcg_at_k, avg_mrr) and raw predictions.
        """
        print(f"Evaluating Top-K Ranking (K={self.top_k})")
        
        # Initializes lists to store per-user metric values.
        metrics = {
            'avg_rank': [],
            'median_rank': [],
            'mrr': [],
            'hit_rate_at_k': [],
            'ndcg_at_k': []
        }
        all_predictions = {} # Dictionary to store raw predictions (item_id, score) for each user.
        
        # Groups test data by user to process each user individually.
        user_groups = self.test_data.groupby('user_id')
        
        # Iterates through each user and their interactions in the test set.
        for user_id, user_interactions in tqdm(user_groups, desc="Evaluating ranking"):
            # Ensures user_id is a string.
            user_id = str(user_id)
            # Extracts test items for the current user. These are considered the "relevant" items for ranking.
            test_items = [str(item) for item in user_interactions['item_id'].tolist()]
            
            try:
                # Collects scores for all test items for the current user.
                item_scores = []
                for item_id in test_items:
                    try:
                        # Calls the recommender's `get_item_score` method.
                        score = self.recommender.get_item_score(str(user_id), str(item_id))
                        item_scores.append((str(item_id), score))
                    except Exception as e:
                        print(f"Error getting score for user {user_id}, item {item_id}: {e}")
                        # Assigns a score of 0.0 if an error occurs for a specific item.
                        item_scores.append((str(item_id), 0.0))
                
                if not item_scores:
                    # Appends 0.0 for all metrics if no scores could be obtained for the user's test items.
                    for metric_list in metrics.values():
                        metric_list.append(0.0)
                    continue
                
                # Stores the unsorted item scores (predictions) before sorting for ranking metrics.
                all_predictions[user_id] = item_scores
                
                # Sorts items by score in descending order to obtain the ranked list.
                item_scores.sort(key=lambda x: x[1], reverse=True)
                ranked_items = [str(item_id) for item_id, _ in item_scores]
                
                # Calculates ranking metrics.
                ranks = []
                # Assigns a rank (1-based index) to each item in the sorted list.
                for i, (item_id, _) in enumerate(item_scores, 1):
                    ranks.append(i)
                
                # Calculates the average and median rank of the test items.
                avg_rank = np.mean(ranks)
                median_rank = np.median(ranks)
                
                # Calculates Mean Reciprocal Rank (MRR). Since all test items are considered relevant,
                # MRR is typically 1 divided by the rank of the highest-ranked test item.
                mrr = 1.0 / ranks[0] if ranks else 0.0 # Handles empty ranks list.
                
                # Calculates Hit Rate@K: the fraction of test items that appear within the top-K recommendations.
                hits_in_top_k = sum(1 for rank in ranks if rank <= self.top_k)
                hit_rate = hits_in_top_k / len(test_items) if test_items else 0.0
                
                # Calculates Normalized Discounted Cumulative Gain (NDCG@K). All test items are treated
                # as equally relevant (relevance score of 1).
                relevant_items_set = set(test_items)
                ndcg = self._calculate_ndcg(ranked_items, relevant_items_set, self.top_k)
                
                # Appends calculated metrics to their respective lists.
                metrics['avg_rank'].append(avg_rank)
                metrics['median_rank'].append(median_rank)
                metrics['mrr'].append(mrr)
                metrics['hit_rate_at_k'].append(hit_rate)
                metrics['ndcg_at_k'].append(ndcg)
                
            except Exception as e:
                print(f"Error evaluating ranking for user {user_id}: {e}")
                # Appends default (worst-case) values for metrics if an error occurs during processing.
                metrics['avg_rank'].append(float('inf'))
                metrics['median_rank'].append(float('inf'))
                metrics['mrr'].append(0.0)
                metrics['hit_rate_at_k'].append(0.0)
                metrics['ndcg_at_k'].append(0.0)
        
        # Aggregates per-user metrics by calculating their means and standard deviations.
        results = {}
        for metric_name, values in metrics.items():
            if values: # Ensures that values list is not empty before calculating statistics.
                if metric_name in ['avg_rank', 'median_rank']:
                    # Filters out infinite values for rank metrics before averaging.
                    finite_values = [v for v in values if np.isfinite(v)]
                    if finite_values:
                        results[f"avg_{metric_name}"] = np.mean(finite_values)
                        results[f"std_{metric_name}"] = np.std(finite_values)
                    else:
                        # Sets to infinity if no finite values are present after filtering.
                        results[f"avg_{metric_name}"] = float('inf')
                        results[f"std_{metric_name}"] = 0.0
                else:
                    results[f"avg_{metric_name}"] = np.mean(values)
                    results[f"std_{metric_name}"] = np.std(values)
            else:
                # Sets default values if no metrics were collected for the entire run.
                results[f"avg_{metric_name}"] = 0.0
                results[f"std_{metric_name}"] = 0.0
        
        # Adds the number of users evaluated and all raw predictions to the results.
        results['num_users_evaluated'] = len(user_groups)
        results['predictions'] = all_predictions 
        
        return results
    
    def _calculate_ndcg(self, ranked_items: List[str], relevant_items: set, k: int) -> float:
        """
        Calculates Normalized Discounted Cumulative Gain at K (NDCG@k) for a single user.

        Args:
            ranked_items: A list of item IDs (string) ranked by the model's scores, in descending order.
            relevant_items: A set of item IDs (string) that are truly relevant to the user.
            k: The number of top ranked items to consider for NDCG calculation.

        Returns:
            The NDCG@k score for the user (float). Returns 0.0 if no relevant items are provided.
        """
        if not relevant_items:
            return 0.0
        
        # Calculates Discounted Cumulative Gain (DCG).
        dcg = 0.0
        # Iterates through the top 'k' ranked items with 1-based indexing for log2.
        for i, item in enumerate(ranked_items[:k], 1):
            if item in relevant_items:
                # Adds relevance score (1.0 for binary relevance) discounted by position.
                dcg += 1.0 / np.log2(i + 1)
        
        # Calculates Ideal Discounted Cumulative Gain (IDCG) for a perfect ranking.
        # It assumes all relevant items are ranked perfectly at the top.
        num_relevant = min(len(relevant_items), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(num_relevant))
        
        # Returns NDCG (DCG / IDCG). Handles division by zero if IDCG is 0.
        return dcg / idcg if idcg > 0 else 0.0


def create_evaluator(task: EvaluationTask, recommender, test_data: pd.DataFrame, 
                    config, **kwargs) -> BaseEvaluator:
    """
    Factory function to create an appropriate evaluator instance based on the specified task.

    This function abstracts the instantiation logic, allowing the caller to
    obtain the correct evaluator type without needing to know the specific
    class names.

    Args:
        task: An `EvaluationTask` enum member specifying the type of evaluation to perform.
        recommender: The recommender system object to be evaluated.
        test_data: A pandas DataFrame containing the test set interactions.
        config: The configuration object.
        **kwargs: Additional keyword arguments to pass to the evaluator's constructor.

    Returns:
        An instance of a concrete `BaseEvaluator` subclass corresponding to the specified task.

    Raises:
        ValueError: If an unknown or unsupported evaluation task is provided.
    """
    
    if task == EvaluationTask.TOP_K_RETRIEVAL:
        return TopKRetrievalEvaluator(
            recommender=recommender,
            test_data=test_data,
            config=config,
            **kwargs
        )
    elif task == EvaluationTask.TOP_K_RANKING:
        return TopKRankingEvaluator(
            recommender=recommender,
            test_data=test_data,
            config=config,
            **kwargs
        )
    else:
        # Raises an error for unsupported evaluation tasks.
        raise ValueError(f"Unknown evaluation task: {task}")


# A mapping from string task names (for backward compatibility or CLI arguments)
# to their corresponding `EvaluationTask` enum members.
TASK_MAPPING = {
    'retrieval': EvaluationTask.TOP_K_RETRIEVAL,
    'ranking': EvaluationTask.TOP_K_RANKING,
    # Explicitly sets removed tasks to None to raise a specific error message.
    'next_item': None,
    'cold_user': None,
    'cold_item': None,
    'beyond_accuracy': None,
    'session_based': None
}


def get_task_from_string(task_name: str) -> EvaluationTask:
    """
    Converts a string representation of an evaluation task into its
    corresponding `EvaluationTask` enum member.

    This utility function is useful for parsing command-line arguments or
    configuration files where task names are typically provided as strings.

    Args:
        task_name: A string representing the name of the evaluation task
                   (e.g., 'retrieval', 'ranking').

    Returns:
        An `EvaluationTask` enum member.

    Raises:
        ValueError: If the provided `task_name` is unknown, unsupported, or
                    corresponds to a task that has been removed.
    """
    # Attempts to map the string task name using the predefined mapping.
    if task_name in TASK_MAPPING:
        task = TASK_MAPPING[task_name]
        if task is None:
            # Raises a specific error if the task is recognized but explicitly removed.
            raise ValueError(f"Task '{task_name}' has been removed in the simplified evaluation framework. "
                           f"Available tasks: {list(EvaluationTask.__members__.keys())}")
        return task
    
    # As a fallback, attempts to directly convert the string to an Enum member.
    try:
        return EvaluationTask(task_name)
    except ValueError:
        # Raises an error if the task name is entirely unrecognized.
        available_tasks = list(EvaluationTask.__members__.keys())
        raise ValueError(f"Unknown task '{task_name}'. Available tasks: {available_tasks}")