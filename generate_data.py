"""
Генерация синтетических данных A/B-теста: тестируем новый дизайн страницы оформления
заказа (checkout) в интернет-магазине. Метрика — конверсия в покупку.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

rng = np.random.default_rng(7)

N_CONTROL = 5200
N_TREATMENT = 5150

TRUE_CR_CONTROL = 0.105     # 10.5% базовая конверсия
TRUE_CR_TREATMENT = 0.124   # 12.4% в тестовой группе (эффект ~+18% relative)

devices = ["mobile", "desktop", "tablet"]
device_p = [0.58, 0.35, 0.07]

start_date = datetime(2026, 5, 1)

def make_group(n, group_name, base_cr, device_effect):
    device = rng.choice(devices, size=n, p=device_p)
    # у разных устройств разная базовая конверсия
    device_mult = np.select(
        [device == "mobile", device == "desktop", device == "tablet"],
        [0.85, 1.25, 0.95]
    )
    day_offset = rng.integers(0, 14, size=n)  # эксперимент шёл 14 дней
    visit_date = [start_date + timedelta(days=int(d)) for d in day_offset]

    # небольшой "novelty effect": в первые дни эффект чуть завышен, потом стабилизируется
    novelty_mult = np.where(np.array(day_offset) < 3, 1.15, 1.0) if device_effect else 1.0

    cr = base_cr * device_mult * novelty_mult
    converted = (rng.uniform(0, 1, n) < cr).astype(int)

    revenue = np.where(
        converted == 1,
        rng.gamma(shape=3.0, scale=18, size=n).clip(5, 400).round(2),
        0.0
    )

    return pd.DataFrame({
        "user_id": [f"U{group_name}{i:06d}" for i in range(n)],
        "group": group_name,
        "device": device,
        "visit_date": visit_date,
        "converted": converted,
        "revenue": revenue,
    })

df_control = make_group(N_CONTROL, "control", TRUE_CR_CONTROL, device_effect=False)
df_treatment = make_group(N_TREATMENT, "treatment", TRUE_CR_TREATMENT, device_effect=True)

df = pd.concat([df_control, df_treatment], ignore_index=True)
df = df.sample(frac=1, random_state=1).reset_index(drop=True)

df.to_csv("./data/ab_test_checkout.csv", index=False)
print(df.shape)
print(df.groupby("group")["converted"].mean())
