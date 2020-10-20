import networkx as nx

import itertools


class Decision:
    """
    A decision is an interaction you may have with the predictions in the
    process of correcting it.

    There are 4 decisions you might make.
    1) TraceDecision:
        - made at every node
        - do you need to augment the prediction with some manual tracing here?
    2) RemoveComponentDecision: If the prediction created a false positive
        - made on every component
        - can we remove this component? Does it contribute nothing?
    3) SplitDecision:
        - made on every edge
        - Does this edge connect two nodes that shouldn't be?
        - Can we remove this edge?
    4) MergeDecision:
        - made on every pair of components
        - Do these components need to be connected?
        - i.e. do they belong to the same neuron?

    It is important that accuracy is monotonically increasing as each
    of these decisions are corrected on the predictions.

    1) TraceDecision:
        - Should strictly increase the accuracy by fixing coverage on all
        false negatives off of this node

    2) RemoveComponentDecision:
        - If this component is removed, accuracy strictly increases since
        there will be one less false positive component

    3) SplitDecision:
        - A corrected split decision should decrease the number of merged
        components by exactly 1. It may increase the number of false positive
        components by 1, leading to a new decision needing to be made? How
        should this work?

    4) MergeDecision:
        - A corrected merge decision should decrease the number of split
        components by exactly 1.
    """


class TraceDecision:
    pass


class RemoveComponentDecision:
    pass


class MergeDecision:
    pass


class FalseMergeDecision:
    pass


def review_error_area_under_curve(gt: nx.Graph, prediction: nx.Graph, config):
    """
    Score a graph given a ground truth graph and a prediction graph and a config.

    We want a score that takes into account the amount of review effort needed
    to get to some desired accuracy.

    Consider a prediction as a set of decision points, where each decision point
    can be correct or incorrect. At every point, the prediction will have an opinion
    on the correct decision. If inspecting these decision points is similarly expensive
    to making the decision, then we want to minimize the number of decisions we must
    inspect.

    First we need to obtain a set of decision points from a prediction.
    A set of decision points must have the property that inspecting/fixing each
    one will leave you with a perfect reconstruction. Fixing a decision point
    should result in a monotonically increasing accuracy. I.e. If you have a
    decision to merge two objects, one of them could be partly correct,
    thus merging them should not increase the number of errors.
    We can just use comatch to count errors in terms of split/merge/fp/fn.
    Fixing an decision should monotonically decrease the number of errors.

    Next we need to calculate accuracy given a prediction and a ground truth.
    """

    matching = match(gt, prediction, config.matching)

    confidence, cost, decision = get_decision_points(prediction, config.decision_points)


def match(gt, prediction, config):
    """
    perform comatch
    """
    return matching


def get_decision_points(prediction: nx.Graph, config):
    """
    Problems: FP
    None of these nodes should be contained. "fixing" a single node might
    split the fp into more false positives, increasing the number of fp
    components. A decision that can be fixed while monotonically improving
    the number of FP components would be to look at the component as a whole
    and determine whether it should be connected or not.
    1 decision per Component?
    Problems: FN
    What is a decision point for fixing a false negative? If every node
    is a FN decision point, number of decision points could get large, and
    how do we assign confidences here?.
    1 decision per Node?
    Problems: Splits/Merges
    1 decision per Edge?
    1 decision per pair of Components?
    """
    # are we missing a branch at this node?
    trace_decisions = []
    for node, attrs in prediction.nodes.items():
        trace_decisions.append(TraceDecision(prediction, node))

    # is this component a false positive component?
    remove_component_decisions = []
    for i, component in enumerate(nx.connected_components(prediction)):
        remove_component_decisions.append(RemoveComponentDecision(prediction, i, component))

    # is this edge a false merge?
    split_decisions = []
    for edge, attrs in prediction.edges.items():
        split_decisions.append(FalseMergeDecision(prediction, edge))

    # Do these two components belong together?
    components = list(enumerate(nx.connected_components(prediction)))
    merge_decisions = []
    for (a, a_component), (b, b_component) in itertools.combinations(components, 2):
        merge_decisions.append(
            MergeDecision(prediction, (a, a_component), (b, b_component))
        )

    return trace_decisions, remove_component_decisions, split_decisions, merge_decisions


def accuracy(gt, pred, matching):
    """
    1 - (# of errors) / (max # of errors)

    This doesn't really make sense to me

    I think accuracy would be the % of gt cable length covered by largest connected
    component of pred
    """
