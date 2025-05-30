# from .models import *
# from sentence_transformers import SentenceTransformer, util
# import torch
# import numpy as np
# import pandas as pd

# model = SentenceTransformer('all-MiniLM-L6-v2')

# def get_user_vector(user):
#     return model.encode(" ".join(user.interest.split(',')), convert_to_tensor=True)

# def semantic_filter(user_vector, products, threshold=0.5):
#     filtered = []
#     for p in products:
#         p_text = f"{p.pdes} {p.category.cname}"
#         p_emb = model.encode(p_text, convert_to_tensor=True)
#         sim = util.cos_sim(user_vector, p_emb).item()
#         if sim > threshold:
#             filtered.append((sim, p))
#     return [p for _, p in sorted(filtered, reverse=True)]

# def build_cf_model():
#     data = []
#     for inter in ProductInteraction.objects.all():
#         if inter.liked:
#             data.append((inter.user.email, inter.product.id, 1))
#         elif inter.disliked:
#             data.append((inter.user.email, inter.product.id, 0))
#     return pd.DataFrame(data, columns=['user', 'item', 'rating'])

# def get_cf_recommendations(user_email, cf_data, top_n=5):
#     from sklearn.metrics.pairwise import cosine_similarity
#     user_item_matrix = cf_data.pivot(index='user', columns='item', values='rating').fillna(0)
#     if user_email not in user_item_matrix.index:
#         return []
#     sim_users = cosine_similarity([user_item_matrix.loc[user_email]], user_item_matrix)[0]
#     sim_scores = dict(zip(user_item_matrix.index, sim_users))
#     sim_scores.pop(user_email, None)

#     scores = {}
#     for u, score in sim_scores.items():
#         rated_items = user_item_matrix.loc[u]
#         for item_id, r in rated_items.items():
#             if user_item_matrix.loc[user_email][item_id] == 0:
#                 scores[item_id] = scores.get(item_id, 0) + score * r
#     top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
#     return [product.objects.get(id=i) for i, _ in top_items]

# def hybrid_recommendation(email):
#     user = profile.objects.get(email=email)
#     all_products = product.objects.all()
#     user_vector = get_user_vector(user)
#     cbf_products = semantic_filter(user_vector, all_products)
#     cf_data = build_cf_model()
#     cf_products = get_cf_recommendations(email, cf_data)

#     # Merge and deduplicate
#     hybrid = list({p.id: p for p in cbf_products + cf_products}.values())
#     return hybrid[:12]
