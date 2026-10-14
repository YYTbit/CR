from .learning import acquisition_reward


def select_message(generator, engine, contexts, count, seed=0, exposure=None,
                   confirmation_repeats=4, final_repeats=8):
    candidates = generator.propose(contexts, count, seed, free=True)
    scored = []
    for message in candidates:
        reading = engine.measure(message, "baseline_selection", 1, 0, exposure)
        if reading is not None:
            payoff = acquisition_reward(engine.game, reading.posterior, reading.action)
            scored.append((payoff, message.identifier, message))
    if not scored:
        return {"status": "additional_measurement_required", "reason": "baseline_candidates_unparseable"}
    scored.sort(key=lambda row: (-row[0], row[1]))
    selected = scored[0][2]
    confirmation = engine.measure(selected, "baseline_confirmation", confirmation_repeats, 0, exposure)
    if confirmation is None:
        return {"status": "additional_measurement_required", "reason": "baseline_confirmation_unparseable"}
    final = engine.measure(selected, "baseline_final", final_repeats, 0, exposure)
    if final is None:
        return {"status": "additional_measurement_required", "reason": "baseline_final_unparseable"}
    return {"status": "ready", "procedure": "prompted_sender" if count == 1 else "best_of_n_payoff",
            "message": selected, "confirmation": confirmation, "final_measurement": final,
            "sender_payoff": acquisition_reward(engine.game, final.posterior, final.action),
            "candidates": len(candidates), "query_account": engine.ledger.summary(),
            "compatible_mixture": False}
