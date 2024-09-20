import numpy as np
import json

################################
# CONSTS
################################

REVIEWS = 0
BOT_ACTION = 1
USER_DECISION = 2
HISTORY_FEATURES = 3


################################

def correct_action(information):
    if information["hotel_value"] >= 8:
        return 1
    else:
        return 0


def random_action(information):
    return np.random.randint(2)


def user_rational_action(information):
    if information["bot_message"] >= 8:
        return 1
    else:
        return 0


def user_picky(information):
    if information["bot_message"] >= 9:
        return 1
    else:
        return 0


def user_sloppy(information):
    if information["bot_message"] >= 7:
        return 1
    else:
        return 0


def user_short_t4t(information):
    if len(information["previous_rounds"]) == 0 \
            or (information["previous_rounds"][-1][BOT_ACTION] >= 8 and
                information["previous_rounds"][-1][REVIEWS].mean() >= 8) \
            or (information["previous_rounds"][-1][BOT_ACTION] < 8 and
                information["previous_rounds"][-1][REVIEWS].mean() < 8):  # cooperation
        if information["bot_message"] >= 8:  # good hotel
            return 1
        else:
            return 0
    else:
        return 0


def user_picky_short_t4t(information):
    if information["bot_message"] >= 9 or ((information["bot_message"] >= 8) and (
            len(information["previous_rounds"]) == 0 or (
            information["previous_rounds"][-1][REVIEWS].mean() >= 8))):  # good hotel
        return 1
    else:
        return 0


def user_hard_t4t(information):
    if len(information["previous_rounds"]) == 0 \
            or np.min(np.array([((r[BOT_ACTION] >= 8 and r[REVIEWS].mean() >= 8)
                                 or (r[BOT_ACTION] <= 8 and r[REVIEWS].mean() < 8)) for r in
                                information["previous_rounds"]])) == 1:  # cooperation
        if information["bot_message"] >= 8:  # good hotel
            return 1
        else:
            return 0
    else:
        return 0


def history_and_review_quality(history_window, quality_threshold):
    def func(information):
        if len(information["previous_rounds"]) == 0 \
                or history_window == 0 \
                or np.min(np.array([((r[BOT_ACTION] >= 8 and r[REVIEWS].mean() >= 8)
                                     or (r[BOT_ACTION] <= 8 and r[REVIEWS].mean() < 8)) for r in
                                    information["previous_rounds"][
                                    -history_window:]])) == 1:  # cooperation from *result's* perspective
            if information["bot_message"] >= quality_threshold:  # good hotel from user's perspective
                return 1
            else:
                return 0
        else:
            return 0
    return func


def topic_based(positive_topics, negative_topics, quality_threshold):
    def func(information):
        review_personal_score = information["bot_message"]
        for rank, topic in enumerate(positive_topics):
            review_personal_score += int(information["review_features"].loc[topic])*2/(rank+1)
        for rank, topic in enumerate(negative_topics):
            review_personal_score -= int(information["review_features"].loc[topic])*2/(rank+1)
        if review_personal_score >= quality_threshold:  # good hotel from user's perspective
            return 1
        else:
            return 0
    return func


def LLM_based(is_stochastic):
    with open(f"data/baseline_proba2go.txt", 'r') as file:
        proba2go = json.load(file)
        proba2go = {int(k): v for k, v in proba2go.items()}

    if is_stochastic:
        def func(information):
            review_llm_score = proba2go[information["review_id"]]
            return int(np.random.rand() <= review_llm_score)
        return func
    else:
        def func(information):
            review_llm_score = proba2go[information["review_id"]]
            return int(review_llm_score >= 0.5)
        return func

def llm_model(generated_scores):
    def func(information):
        review_generated_scores = generated_scores[information["review_id"]]
        return int(review_generated_scores >= 8)
    return func

def history_and_llm(history_window, quality_threshold, generated_scores, use_statistics):
    def func(information):

        # If 3 rounds have not yet passed, determine whether to go,
        # based on the average score between the score of the bot and the score of the llm model
        avg_score = (information["bot_message"] + generated_scores[information["review_id"]]) / 2
        if len(information["previous_rounds"]) < 3 \
                or history_window == 0 \
                or np.min(np.array([((r[BOT_ACTION] >= 8 and r[REVIEWS].mean() >= 8)
                                     or (r[BOT_ACTION] <= 8 and r[REVIEWS].mean() < 8)) for r in
                                    information["previous_rounds"][-history_window:]])) == 1:
            return int(avg_score >= quality_threshold)

        # if dm identifies an action pattern of the bot based on the dm decision whether to go to the hotel or not,
        # if it chose to go in the previous round the bot returns a score lower than or equal to the average
        # otherwise the bot returns a score greater than or equal to the average.
        elif np.min([((r[HISTORY_FEATURES]["last_didGo_True"] and r[BOT_ACTION] <= r[REVIEWS].mean()) or
                      (r[HISTORY_FEATURES]['last_didGo_False'] and r[BOT_ACTION] >= r[REVIEWS].mean())) for r in
                     information["previous_rounds"][1:]]) == 1:
            if information["previous_rounds"][-1][HISTORY_FEATURES]['didGo'] and information["bot_message"] >= 8:
                return 1
            elif not information["previous_rounds"][-1][HISTORY_FEATURES]['didGo'] and information["bot_message"] < 8:
                return 0

        # dm identifies an action pattern of the bot based on the dm wining history
        elif np.min([((r[HISTORY_FEATURES]['last_didWin_True'] and r[BOT_ACTION] <= r[REVIEWS].mean()) or (
                r[HISTORY_FEATURES]['last_didWin_False'] and r[BOT_ACTION] >= r[REVIEWS].mean())) for r in
                     information["previous_rounds"][1:]]) == 1:
            if information["previous_rounds"][-1][HISTORY_FEATURES]['didWin'] and information["bot_message"] >= 8:
                return 1
            elif not information["previous_rounds"][-1][HISTORY_FEATURES]['didWin'] and information["bot_message"] < 8:
                return 0

         #dm identifies an action pattern of the bot based on the user_points/bot_points
        elif np.min([((r[HISTORY_FEATURES]['user_points'] > r[HISTORY_FEATURES]['bot_points'] and
                       r[BOT_ACTION] <= r[REVIEWS].mean())
                      or (r[HISTORY_FEATURES]['bot_points'] >= r[HISTORY_FEATURES]['user_points'] and
                          r[BOT_ACTION] >= r[REVIEWS].mean())) for r in
                     information["previous_rounds"][1:]]) == 1:
            if information["previous_rounds"][-1][HISTORY_FEATURES]['user_points'] > \
                    information["previous_rounds"][-1][HISTORY_FEATURES]['bot_points']:
                if information["bot_message"] >= 8:
                    return 1
            elif information["previous_rounds"][-1][HISTORY_FEATURES]['user_points'] < \
                    information["previous_rounds"][-1][HISTORY_FEATURES]['bot_points']:
                if information["bot_message"] < 8:
                    return 0

        if use_statistics:
            median_of_max_dist, median_of_min_dist, median_of_means = 1.3915966386554617, 2.538095238095236, 8.300595238095237
            if (information["bot_message"] - median_of_max_dist) >= 8:
                return 1
            elif information["bot_message"] + median_of_min_dist <= 8:
                return 0
            elif information["bot_message"] >= median_of_means:
                return 1
            else:
                return 0
        return np.random.randint(2)

    return func


