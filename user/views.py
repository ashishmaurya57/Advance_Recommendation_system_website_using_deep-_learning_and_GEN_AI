from django.shortcuts import render
from .models import *
from django.http import HttpResponse
import datetime
from django.db import connection
import numpy as np
from django.shortcuts import redirect, get_object_or_404
from .models import ProductInteraction, product
from django.http import JsonResponse, Http404
from django.contrib.auth.decorators import login_required
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from textblob import TextBlob
from collections import defaultdict
from django.core.cache import cache
from textblob import TextBlob

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
import numpy as np
from django.shortcuts import render
from .models import category, product, addtocart, profile
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
chatgroq = ChatGroq(api_key=api_key, model="llama3-70b-8192", temperature=0.5)


def home(req):
    cdata = category.objects.all().order_by('-id')[:6]
    pdata = product.objects.all().order_by('-id')[:12]
    noofitemsincart = addtocart.objects.filter(userid=req.session.get('userid')).count()

    return render(req, 'user/index.html', {
        "data": cdata,
        "products": pdata,
        "noofitemsincart": noofitemsincart,
    })

def get_recommendations_api(req):
    if 'userid' not in req.session:
        return JsonResponse({'error': 'Not logged in'}, status=401)

    try:
        user_profile = profile.objects.get(email=req.session['userid'])
        recommended_books = recommend_products(user_profile)  # returns cached or computes

        data = [{
            'id': p.id,
            'name': p.name,
            'desc': p.pdes,
            'image': p.ppic.url if p.ppic else '',
            'language': p.language,
            'category': p.category.cname,
            'tprice': p.tprice,
            'disprice': p.disprice,
            'ppic': p.ppic.url if p.ppic else ''
        } for p in recommended_books]

        return JsonResponse({'products': data})

    except profile.DoesNotExist:
        return JsonResponse({'products': []})

    
INTERACTION_THRESHOLD = 0.2

from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# Load models once (do this at the start of your app)
sentiment_analyzer = pipeline("sentiment-analysis")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_sentiment_score(text):
    """
    Returns a sentiment score between 0 and 1.
    1 = very positive sentiment, 0 = very negative sentiment.
    """
    result = sentiment_analyzer(text[:512])[0]  # limit length for performance
    if result['label'] == 'POSITIVE':
        return result['score']
    else:
        return 1 - result['score']

def get_semantic_similarity(text1, text2):
    """
    Returns cosine similarity between two texts (0 to 1).
    """
    emb1 = semantic_model.encode(text1, convert_to_tensor=True)
    emb2 = semantic_model.encode(text2, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2).item()

def get_interaction_score(user_profile, p):
    interactions = UserInteraction.objects.filter(user=user_profile, product=p)

    views = interactions.filter(interaction_type='view').count()
    clicks = interactions.filter(interaction_type='click').count()
    add_to_cart = interactions.filter(interaction_type='add_to_cart').count()
    ratings = interactions.filter(interaction_type='rating').values_list('rating', flat=True)

    
    rating_score = 0
    ratings = [float(r) for r in ratings if r is not None]
    if ratings:
        rating_score = sum(ratings) / len(ratings)  

    score = (
        0.2 * min(views, 10)+        # Up to 0.3
        0.1 * min(clicks, 10)+         # Up to 0.2
        0.5 * min(add_to_cart, 2) +    # Up to 0.2
        0.4 * rating_score                 # Up to 0.3
    )

    print(f"product: {p}")
    print(f"rating : {round(rating_score, 2)}")
    print(f"score : {round(score, 2)}")

    return round(score, 2)
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

prompt = PromptTemplate(
    input_variables=["user_interests", "product_description", "category_name"],
    template="""
You are a recommendation engine.

Input:
- User Interests: {user_interests}
- Product Description: {product_description}
- Category: {category_name}

Instructions:
1. Treat 'user_interests' as a list of keyword phrases.
2. Process each interest **one at a time**, in order.
   a. Compare it with product_description and category_name.
   b. Match based on keyword overlap and sentiment/contextual alignment.
   c. Calculate a relevance score between 0.0 (not relevant) and 1.0 (highly relevant).
3. If any interest yields a score **greater than or equal to 0.8**, immediately return that score and stop further checks.
4. If none reach that threshold, return the **highest score** found.
5. **Important: Return only the score as a float. No explanation, no extra text. Example: 0.75**
"""
)


chain = LLMChain(llm=chatgroq, prompt=prompt)


# Weights for interaction types
INTERACTION_WEIGHTS = {
    "view": 1,
    "click": 2,
    "add_to_cart": 3,
    "rating": 4  # You could scale by rating value if needed
}

import re
from concurrent.futures import ThreadPoolExecutor, as_completed

def recommend_products(user_profile, force_update=False):
    print("🔥 recommend_products called")
    cache_key = f"recommendations_combined_{user_profile.email}"
    cached = cache.get(cache_key)
    if cached and not force_update:
        # Return cached recommendations immediately
        return cached

    user_tags = [tag.name.lower() for tag in user_profile.interests.all()]
    user_tags_set = set(user_tags)
    # Load or initialize LLM cache - product ids with their scores
    llm_cache_key = f"llm_recommendations_{user_profile.email}"
    llm_cache = cache.get(llm_cache_key) or {}
    # Load all products with only necessary fields and related category to reduce DB overhead
    all_products = product.objects.select_related('category').only('id', 'pdes', 'category__cname')

    # 1. Precompute interaction scores for all products once
    def max_similarity_with_interests(user_tags, product_desc):
        scores = [get_semantic_similarity(tag, product_desc) for tag in user_tags]
        return max(scores) if scores else 0.0

    # Step 2: Filter products based on semantic similarity before LLM call
    likely_relevant_products = [
            p for p in all_products
            if max_similarity_with_interests(user_tags, p.pdes.lower()) > 0.4
            or max_similarity_with_interests(user_tags, p.category.cname.lower()) > 0.5
        ]


    # 3. Compute LLM scores in parallel to boost speed
    llm_scores = {}

    def fetch_llm_score(p):
        if p.id in llm_cache:
            # Use cached score
            cached_score = llm_cache[p.id]
            if cached_score >= 0.8:
                return (p.id, (p, cached_score))
            else:
                return None
        try:
            result = chain.invoke({
                "user_interests": user_tags,
                "product_description": p.pdes.lower(),
                "category_name": p.category.cname.lower()
            })
            match = re.search(r"[-+]?\d*\.\d+|\d+", str(result))
            if match:
                score = float(match.group())
                llm_cache[p.id] = score
                if score >= 0.8:
                    return (p.id, (p, score))
            return None
        except Exception as e:
            print(f"⚠️ Error invoking LLM for product {p.id}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch_llm_score, p) for p in likely_relevant_products]
        for future in as_completed(futures):
            res = future.result()
            if res:
                pid, data = res
                llm_scores[pid] = data
    cache.set(llm_cache_key, llm_cache, timeout=1800)
    # 4. Build interaction-based recommendations with semantic similarity and sentiment filters
    recommendations_interaction = {}
    for p in all_products:
        score = get_interaction_score(user_profile, p)
        if score > 1.6:
            product_desc = p.pdes.lower()
            category_name = p.category.cname.lower()
            liked_sentiment = get_sentiment_score(product_desc)

            for candidate_product in all_products:
                if candidate_product.id == p.id:
                    continue
                candidate_desc = candidate_product.pdes.lower()
                candidate_cat = candidate_product.category.cname.lower()
                candidate_sentiment = get_sentiment_score(candidate_desc)

                cat_match = 1 if category_name == candidate_cat else 0
                sim_score = get_semantic_similarity(product_desc, candidate_desc)

                if cat_match and sim_score >= 0.7 and candidate_sentiment >= 0.7:
                    combined_score = 0.4 * score + 0.4 * sim_score + 0.2 * candidate_sentiment

                    if candidate_product not in recommendations_interaction or recommendations_interaction[candidate_product] < combined_score:
                        recommendations_interaction[candidate_product] = combined_score

    # Add original products with high interaction score if not already present
    for p in all_products:
        if get_interaction_score(user_profile, p) > 1.5:
            if p not in recommendations_interaction:
                recommendations_interaction[p] = get_interaction_score(user_profile, p)


    # 5. Combine recommendations: intersection, LLM-only, and interaction-only
    interaction_products = set(p.id for p in recommendations_interaction.keys())
    llm_products = set(llm_scores.keys())

    intersection_ids = interaction_products & llm_products
    llm_only_ids = llm_products - intersection_ids
    interaction_only_ids = interaction_products - intersection_ids

    recommended = []

    # Intersection: sorted by LLM score desc
    intersection_sorted = sorted(intersection_ids, key=lambda pid: -llm_scores[pid][1])
    recommended += [llm_scores[pid][0] for pid in intersection_sorted]

    # LLM-only: sorted by LLM score desc
    llm_only_sorted = sorted(llm_only_ids, key=lambda pid: -llm_scores[pid][1])
    recommended += [llm_scores[pid][0] for pid in llm_only_sorted]

    # Interaction-only: sorted by interaction combined score desc
    interaction_only_sorted = sorted(
        interaction_only_ids,
        key=lambda pid: -recommendations_interaction[next(prod for prod in recommendations_interaction if prod.id == pid)]
    )
    recommended += [next(prod for prod in recommendations_interaction if prod.id == pid) for pid in interaction_only_sorted]

    # Cache results for 30 minutes
    cache.set(cache_key, recommended, timeout=1800)
    return recommended
import threading
from django.core.cache import cache
def async_update_recommendations(user_profile):
    def update():
        recommend_products(user_profile, force_update=True)
    threading.Thread(target=update).start()
# def recommend_products(user_profile):
#     print("🔥 recommend_products called")
#     cache_key = f"recommendations_combined_{user_profile.email}"
#     cached = cache.get(cache_key)
#     if cached:
#         return cached

#     # Step 1: User Interests (for LLM)
#     user_tags = [tag.name.lower() for tag in user_profile.interests.all()]
    
#     user_tags_str = " ".join(user_tags)

#     all_products = product.objects.all()
#     recommendations_interaction = {}

#     # Step 2: Interaction-based recommendations
#     for p in all_products:
#         score = get_interaction_score(user_profile, p)
#         if score > 1.6:
#             product_desc = p.pdes.lower()
#             category_name = p.category.cname.lower()
#             liked_sentiment = get_sentiment_score(product_desc)

#             for candidate_product in all_products:
#                 if candidate_product.id == p.id:
#                     continue
                
#                 candidate_desc = candidate_product.pdes.lower()
#                 candidate_cat = candidate_product.category.cname.lower()
#                 candidate_sentiment = get_sentiment_score(candidate_desc)

#                 cat_match = 1 if category_name == candidate_cat else 0
#                 sim_score = get_semantic_similarity(product_desc, candidate_desc)
#                 # print(f"matching-{cat_match}")
#                 # print(f"sim_score-{sim_score}")
#                 # print(f"candidate_Sentimen-{candidate_sentiment}")
#                 if cat_match and sim_score >= 0.7 and candidate_sentiment >= 0.7:
#                     combined_score = 0.4 * score + 0.4 * sim_score + 0.2 * candidate_sentiment
#                     # print(f"combined score-- {combined_score}")

#                     # Keep highest combined score per candidate
#                     if candidate_product not in recommendations_interaction or recommendations_interaction[candidate_product] < combined_score:
#                         recommendations_interaction[candidate_product] = combined_score

#     # Add original products with high interaction score (in case they were missed)
#     for p in all_products:
#         if get_interaction_score(user_profile, p) > 1.5:
#             if p not in recommendations_interaction:
#                 recommendations_interaction[p] = get_interaction_score(user_profile, p)

#     # Step 3: LLM-based recommendation scores
#     llm_scores = {}
#     for p in all_products:
#             # print(f"user_interes: {user_tags}")
            
#             result = chain.invoke({
#                 "user_interests": user_tags,
#                 "product_description": p.pdes.lower(),
#                 "category_name": p.category.cname.lower()
#             })

#             # Safely extract float from LLM output
#             try:
#                 score = float(result)
#             except (ValueError, TypeError):
#                 import re
#                 match = re.search(r"[-+]?\d*\.\d+|\d+", str(result))
#                 if match:
#                     score = float(match.group())
#                 else:
#                     print(f"⚠️ Could not parse score from result: {result}")
#                     continue  # Skip this product if parsing fails

#             if score >= 0.8:
#                 llm_scores[p.id] = (p, score)


#     # Step 4: Combine interaction and LLM recommendations

#     interaction_products = set(p.id for p in recommendations_interaction.keys())
#     llm_products = set(llm_scores.keys())

#     intersection_ids = interaction_products & llm_products
#     llm_only_ids = llm_products - intersection_ids
#     interaction_only_ids = interaction_products - intersection_ids

#     recommended = []

#     # 1. Books common in both (LLM + interaction) → sorted by LLM score desc
#     intersection_sorted = sorted(intersection_ids, key=lambda pid: -llm_scores[pid][1])
#     recommended += [llm_scores[pid][0] for pid in intersection_sorted]

#     # 2. Books with only LLM match → sorted by LLM score desc
#     llm_only_sorted = sorted(llm_only_ids, key=lambda pid: -llm_scores[pid][1])
#     recommended += [llm_scores[pid][0] for pid in llm_only_sorted]

#     # 3. Books with only interaction match → sorted by interaction combined score desc
#     interaction_only_sorted = sorted(interaction_only_ids, key=lambda p: -recommendations_interaction[next(prod for prod in recommendations_interaction if prod.id == p)])
#     recommended += [next(prod for prod in recommendations_interaction if prod.id == pid) for pid in interaction_only_sorted]

#     # Step 5: Cache and return
#     cache.set(cache_key, recommended, timeout=1800)
#     return recommended

# def recommend_products(user_profile):
#     print("🔥 recommend_products called")
#     cache_key = f"recommendations_{user_profile.email}"
#     cached = cache.get(cache_key)
#     if cached:
#         return cached

#     all_products = product.objects.all()
#     interacted_products = []
    
#     # Phase 1: Filter by Interaction Score
#     for p in all_products:
#         score = get_interaction_score(user_profile, p)
#         if score >= INTERACTION_THRESHOLD:
#             interacted_products.append((p, score))

#     print(f"📚 Positively interacted books: {len(interacted_products)}")

#     # Setup LLM prompt
#     prompt_template = """
# You are a recommendation engine. Evaluate relevance of another product based on the user's interest and the context of a liked product.

# User Interests: {user_interests}  
# Liked Product Description: {liked_description}  
# Liked Category: {liked_category}  
# Candidate Product Description: {product_description}  
# Candidate Category: {category_name}  

# Instructions:  
# 1. Match user interests and the liked product with the candidate product.  
# 2. Return a float between 0.0 and 1.0 indicating how closely the candidate matches.  
# 3. No text, just a number.
# """
#     prompt = PromptTemplate(
#         input_variables=["user_interests", "liked_description", "liked_category", "product_description", "category_name"],
#         template=prompt_template,
#     )
#     chain = LLMChain(llm=chatgroq, prompt=prompt)

#     user_tags = [tag.name.lower() for tag in user_profile.interests.all()]
#     user_tags_str = " ".join(user_tags)

#     final_recommendations = []

#     for liked_product, inter_score in interacted_products:
#         liked_desc = liked_product.pdes.lower()
#         liked_category = liked_product.category.cname.lower()

#         for candidate in all_products:
#             if candidate.id == liked_product.id:
#                 continue

#             # Match by same category
#             if candidate.category.cname.lower() != liked_category:
#                 continue

#             # Match positive sentiment
#             sentiment = TextBlob(candidate.pdes.lower()).sentiment.polarity
#             if sentiment < 0.2:
#                 continue

#             # Text similarity
#             tfidf = TfidfVectorizer(stop_words="english")
#             vectors = tfidf.fit_transform([liked_desc, candidate.pdes.lower()])
#             text_sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

#             # LLM Score
#             try:
#                 llm_score = float(chain.run(
#                     user_interests=user_tags_str,
#                     product_description=candidate.pdes.lower(),
#                     category_name=candidate.category.cname.lower()
#                 ).strip())
#                 llm_score = max(0.0, min(1.0, llm_score))
#             except:
#                 llm_score = 0.0

#             final_score = (
#                 0.4 * llm_score +
#                 0.3 * text_sim +
#                 0.2 * ((sentiment + 1) / 2) +
#                 0.1 * inter_score  # use source product interaction as signal
#             )
#             print(final_score)

#             final_recommendations.append((candidate, round(final_score, 3)))

#     # Sort by score
    
#     final_recommendations.sort(key=lambda x: x[1], reverse=True)
#     unique_recommended = []
#     seen_ids = set()
#     for book, score in final_recommendations:
#         if book.id not in seen_ids:
#             seen_ids.add(book.id)
#             unique_recommended.append(book)

#     cache.set(cache_key, unique_recommended[:10], timeout=1800)
#     return unique_recommended[:10]   

# def recommend_products(user_profile):
#     print("🔥 recommend_products called")
#     cache_key = f"recommendations_{user_profile.email}"  # Use email instead of id
#     cached = cache.get(cache_key)
#     if cached:
#         return cached

#     user_tags = [tag.name.lower() for tag in user_profile.interests.all()]
#     print("User tags:", user_tags)  
#     user_tags_str = " ".join(user_tags)
#     all_products = product.objects.all()
#     filtered_products = []

#     # Define LangChain prompt
#     prompt_template = """
# You are a recommendation engine. Evaluate the relevance of a product based on the user's interests.

# User Interests: {user_interests}  
# Product Description: {product_description}  
# Category: {category_name}  

# Instructions:  
# 1. Check if any keywords from the user's interests appear in the product description or category.  
# 2. Check if the overall sentiment and context of the product aligns positively with the user's interests.  
# 3. If either condition is satisfied, return **only "1"**.  
# 4. If neither condition is met, return **only "0"**.  

# Respond with a single digit: "1" for relevant, or "0" for not relevant. Do not include any explanation.
# """

#     prompt = PromptTemplate(
#         input_variables=["user_interests", "product_description", "category_name"],
#         template=prompt_template,
#     )
#     chain = LLMChain(llm=chatgroq, prompt=prompt)

#     for p in all_products:
#         product_desc = p.pdes.lower()
#         category_name = p.category.cname.lower()

#         # Run LLMChain
#         result = chain.run(
#             user_interests=user_tags_str,
#             product_description=product_desc,
#             category_name=category_name
#         ).strip()

#         print("LLM result:", result.strip())  # debug output

#         if result == "1":
#             filtered_products.append(p)

#     filtered_top = filtered_products
#     cache.set(cache_key, filtered_top, timeout=1800)  # Cache for 30 mins
#     return filtered_top



def about(req):
    noofitemsincart = addtocart.objects.all().count()
    return render(req, 'user/about.html', {"noofitemsincart":noofitemsincart})


def contactus(request):
    noofitemsincart = addtocart.objects.all().count()
    status = False
    if request.method == 'POST':
        Name = request.POST.get("name", "")
        Mobile = request.POST.get("mobile", "")
        Email = request.POST.get("email", "")
        Message = request.POST.get("msg", "")
        x = contact(name=Name, email=Email, mobile=Mobile, message=Message)
        x.save()
        status = True
        # return HttpResponse("<script>alert('Thanks for enquiry...');window.location.href='/user/contactus/'</script>")

    return render(request, 'user/contactus.html', {'S': status,"noofitemsincart":noofitemsincart})


def services(req):
    return render(req, 'user/services.html')


def myorders(request):
    userid = request.session.get('userid')
    oid = request.GET.get('oid')
    noofitemsincart = addtocart.objects.all().count()
    orderdata = ""
    if userid:
        cursor = connection.cursor()
        cursor.execute(
            "select o.*,p.* from user_order o,user_product p where o.pid=p.id and o.userid='" + str(userid) + "'")
        orderdata = cursor.fetchall()
        if oid:
            result = order.objects.filter(id=oid, userid=userid)
            result.delete()
            return HttpResponse(
                "<script>alert('your order has been cancelled');window.location.href='/user1/myorders'</script>")

    return render(request, 'user/myorders.html', {"pendingorder": orderdata ,"noofitemsincart":noofitemsincart})

def myprofile(request):
    user = request.session.get('userid')  # Get the logged-in user's email from the session
    pdata = profile.objects.filter(email=user).first()  # Fetch the user's profile
    noofitemsincart = addtocart.objects.all().count()
    print(pdata.email)
    interests_list = []
    # if pdata:
    #     # Split the interests field into a list
    #     # interests_list = pdata.interests.split(', ') if pdata.interests else []
    # else:
    #     interests_list = []

    if user:
        if request.method == 'POST':
            # Get form data
            name = request.POST.get("name", "")
            DOB = request.POST.get("dob", "").strip()  # Get and strip the DOB field
            mobile = request.POST.get("mobile", "")
            password = request.POST.get("passwd", "")
            address = request.POST.get("address", "")
            selected_genres = request.POST.get("interests", "")  # Get the comma-separated genres

            # Validate DOB
            if DOB:
                try:
                    # Ensure the DOB is in the correct format
                    datetime.datetime.strptime(DOB, "%Y-%m-%d")
                    pdata.dob = DOB
                except ValueError:
                    return HttpResponse(
                        "<script>alert('Invalid date format. Please use YYYY-MM-DD.');window.location.href='/user1/myprofile/';</script>"
                    )
            else:
                pdata.dob = None  # Set to None if DOB is empty

            # Update profile fields
            pdata.name = name
            pdata.mobile = mobile
            pdata.passwd = password
            pdata.address = address
            pdata.interests = selected_genres  # Save the updated interests

            # Update profile picture if provided
            if 'ppic' in request.FILES:
                pdata.ppic = request.FILES['ppic']

            # Save the updated profile
            pdata.save()

            return HttpResponse(
                "<script>alert('Your profile updated successfully..');window.location.href='/user1/myprofile/'</script>"
            )

    return render(request, 'user/myprofile.html', {
        "profile": pdata,
        "noofitemsincart": noofitemsincart,
        "interests_list": interests_list,
    })

def prod(request):
    cdata = category.objects.all().order_by('-id')
    noofitemsincart = addtocart.objects.all().count()
    x = request.GET.get('abc')
    # pdata=""
    if x is not None:
        pdata = product.objects.filter(category=x)
    else:
        pdata = product.objects.all().order_by('-id')
    return render(request, 'user/products.html', {"cat": cdata, "products": pdata ,"noofitemsincart":noofitemsincart})


def signup(req):
    noofitemsincart = addtocart.objects.all().count()
    if req.method == "POST":
        name = req.POST.get("name", "")
        DOB = req.POST.get("dob", "")
        email = req.POST.get("email", "")
        mobile = req.POST.get("mobile", "")
        password = req.POST.get("passwd", "")
        address = req.POST.get("address", "")
        picname = req.FILES['ppic']
        interests_raw = req.POST.get("interests", "")  # Comma-separated string

        d = profile.objects.filter(email=email)
        if d.exists():
            return HttpResponse("<script>alert('Already registered..');window.location.href='/user1/signup/'</script>")
        else:
            user_profile = profile(
                name=name,
                dob=DOB,
                mobile=mobile,
                email=email,
                passwd=password,
                address=address,
                ppic=picname
            )
            user_profile.save()

        interest_list = [i.strip() for i in interests_raw.split(",") if i.strip()]
        interests_objs = []

        for interest_name in interest_list:
            interest_obj, _ = InterestTag.objects.get_or_create(name=interest_name)
            interests_objs.append(interest_obj)

        user_profile.interests.set(interests_objs)

        return HttpResponse("<script>alert('Registered successfully..');window.location.href='/user1/signup/'</script>")

    return render(req, 'user/signup.html', {"noofitemsincart": noofitemsincart})






def signin(req):
    noofitemsincart = addtocart.objects.all().count()
    if req.method == 'POST':
        uname = req.POST.get('email', "")
        pwd = req.POST.get('passwd', "")
        checkuser = profile.objects.filter(email=uname, passwd=pwd)
        if (checkuser):
            req.session["userid"] = uname

            return HttpResponse(
                "<script>alert('Logged In Successfully..');window.location.href='/user1/signin/';</script>")
        else:
            return HttpResponse(
                "<script>alert('User Id or Password is incorrect');window.location.href='/user1/signin/';</script>")
    return render(req, 'user/signin.html',{"noofitemsincart":noofitemsincart})



from .models import UserInteraction  # make sure imported
def log_interaction(user, product, interaction_type,user_profile):
    # user_profile = profile.objects.get(email=request.session['userid'])
    async_update_recommendations(user_profile)
    interaction, created = UserInteraction.objects.get_or_create(
        user=user,
        product=product,
        interaction_type=interaction_type,
        defaults={'count': 1}
    )
    if not created and interaction_type in ['view', 'click']:
        interaction.count += 1
        interaction.timestamp = timezone.now()
        interaction.save()

def log_rating_interaction(user, product, new_rating, user_profile):
    # user_profile = profile.objects.get(email=request.session['userid'])
    async_update_recommendations(user_profile)
    interaction, created = UserInteraction.objects.get_or_create(
        user=user,
        product=product,
        interaction_type='rating',
        defaults={'rating': new_rating}
    )
    if not created:
        interaction.rating = round((interaction.rating + new_rating) / 2, 2)
        interaction.timestamp = timezone.now()
        interaction.save()
 # or wherever you place those functions

def viewdetails(request):
    noofitemsincart = addtocart.objects.all().count()
    product_id = request.GET.get('msg')

    if product_id:
        product_id = product_id.rstrip('/')

    product_obj = product.objects.filter(id=product_id).first()
    if not product_obj:
        return HttpResponse("<script>alert('Product not found.');window.location.href='/user1/home/';</script>")

    reviews = Review.objects.filter(product=product_obj).order_by('-created_at')

    # Log view
    if request.session.get('userid'):
        user = profile.objects.get(email=request.session['userid'])
        log_interaction(user, product_obj, 'view', user)

    if request.method == "POST":
        if request.session.get('userid'):
            user = profile.objects.get(email=request.session['userid'])
            rating = int(request.POST.get('rating', 1))
            comment = request.POST.get('comment', '')

            Review.objects.create(product=product_obj, user=user, rating=rating, comment=comment)

            log_rating_interaction(user, product_obj, rating, user)

            return HttpResponse(
                f"<script>alert('Your review has been submitted successfully.');window.location.href='/user1/viewdetails?msg={product_id}/';</script>"
            )
        else:
            return HttpResponse(
                "<script>alert('You need to log in to submit a review.');window.location.href='/user1/signin/';</script>"
            )
    languages = ["english", "hindi", "spanish", "french", "german", "chinese", "japanese", "russian", "arabic", "portuguese"]   
    original_language = product_obj.language  # Assuming this is stored
    
 
    
    return render(request, 'user/viewdetails.html', {
        'languages': languages,
        "product": product_obj,
        "reviews": reviews,
        "noofitemsincart": noofitemsincart,
        'original_language': original_language,
    })
# from .utils import log_interaction

def process(request):
    userid = request.session.get('userid')
    pid = request.GET.get('pid')
    btn = request.GET.get('bn')
    print(userid, pid, btn)

    if userid is not None:
        user = profile.objects.get(email=userid)
        prod = product.objects.get(id=pid)

        log_interaction(user, prod, 'click', user)  # Track click

        if btn == 'cart':
            checkcartitem = addtocart.objects.filter(pid=pid, userid=userid)
            if checkcartitem.count() == 0:
                addtocart(pid=pid, userid=userid, status=True, cdate=datetime.datetime.now()).save()
            else:
                return HttpResponse(
                    "<script>alert('Your item is successfully added to cart..');window.location.href='/user1/cart/'</script>")

        elif btn == 'order':
            order(pid=pid, userid=userid, remarks="pending", status=True, odate=datetime.datetime.now()).save()
            return HttpResponse(
                "<script>alert('your order has confirmed...');window.location.href='/user1/myorders/'</script>")

        elif btn == 'orderfromcart':
            res = addtocart.objects.filter(pid=pid, userid=userid)
            res.delete()
            order(pid=pid, userid=userid, remarks="pending", status=True, odate=datetime.datetime.now()).save()
            return HttpResponse(
                "<script>alert('your order has confirmed...');window.location.href='/user1/myorders/'</script>")

        return render(request, 'user/process.html', {"alreadylogin": True})
    else:
        return HttpResponse("<script>window.location.href='/user1/signin/'</script>")


def logout(request):
    del request.session['userid']
    # return render(request,'user/logout.html')
    return HttpResponse("<script>window.location.href='/user1/home/'</script>")


def cart(request):
    noofitemsincart = addtocart.objects.all().count()
    cartdata={}
    if request.session.get('userid'):
        userid = request.session.get('userid')
        cursor = connection.cursor()
        cursor.execute("select c.*,p.* from user_addtocart c,user_product p where p.id=c.pid")
        cartdata = cursor.fetchall()
        pid = request.GET.get('pid')
        if request.GET.get('pid'):
            res = addtocart.objects.filter(id=pid, userid=userid)
            res.delete()
            return HttpResponse(
                "<script>alert('Your product has been removed successfully');window.location.href='/user1/cart/'</script>")

    return render(request, 'user/cart.html',{"cart":cartdata,"noofitemsincart":noofitemsincart})


from collections import Counter
from .models import SearchLog, profile

def update_user_interests(user):
    # Get all search queries for the user
    searches = SearchLog.objects.filter(user=user).values_list('query', flat=True)
    
    # Count the frequency of each search term
    search_counts = Counter(searches)
    
    # Get the top 5-10 most common search terms
    top_interests = [term for term, _ in search_counts.most_common(10)]
    
    # Update the user's interests field
    user.interests = ", ".join(top_interests)
    user.save()

from django.shortcuts import render
from transformers import pipeline
from torch.nn.functional import softmax
import torch
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from .models import product, category, profile, SearchLog, InterestTag

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = stopwords.words('english')
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Load RoBERTa sentiment model
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

def preprocess_text(text):
    text = text.lower()
    words = text.split()
    processed_words = [word for word in words if word not in stop_words]
    return " ".join(processed_words)

def advanced_sentiment_analysis(text):
    try:
        result = sentiment_pipeline(text.strip())[0]
        label_map = {
            'LABEL_0': 'negative',
            'LABEL_1': 'neutral',
            'LABEL_2': 'positive'
        }
        return label_map.get(result['label'])
    except:
        return None

def search_view(request):
    user_profile = profile.objects.get(email=request.session['userid'])
    query = request.GET.get('q', '').strip()
    query_lower = query.lower()
    user_email = request.session.get('userid')
    pdata = profile.objects.filter(email=user_email).first()
    if 'book' in query_lower:
        # Split query into words, filter out 'book', then rejoin
        filtered_words = [word for word in query_lower.split() if word != 'book']
        query = " ".join(filtered_words)
    if 'science' in query_lower:
        # Split query into words, filter out 'book', then rejoin
        filtered_words = [word for word in query_lower.split() if word != 'book']
        query = " ".join(filtered_words)
    

    products = product.objects.all()
    if not products:
        return render(request, 'user/search_results.html', {'message': 'No products is found.'})

    # Fuzzy match for exact title (threshold 90)
    for p in products:
        if fuzz.ratio(p.name.lower(), query_lower) > 80:
            if request.user.is_authenticated and pdata:
                cat_name = p.category.cname
                tag_obj, _ = InterestTag.objects.get_or_create(name=cat_name)
                if tag_obj not in pdata.interests.all():
                    pdata.interests.add(tag_obj)
                    pdata.save()
            return render(request, 'user/search_results.html', {'products': [p], 'query': query})

    # Fuzzy match for publisher
    matched_publisher = None
    for p in products:
        if p.publisher and fuzz.partial_ratio(p.publisher.lower(), query_lower) > 85:
            matched_publisher = p.publisher
            break
    if matched_publisher:
        products = products.filter(publisher__icontains=matched_publisher)
        return render(request, 'user/search_results.html', {'products': products, 'query': query})

    # Check for category match
    all_categories = category.objects.values_list('cname', flat=True)
    matched_category = next((c for c in all_categories if c.lower() in query_lower), None)
    if matched_category:
        filtered_products = products.filter(category__cname__iexact=matched_category)
        if request.user.is_authenticated and pdata:
            tag_obj, _ = InterestTag.objects.get_or_create(name=matched_category)
            if tag_obj not in pdata.interests.all():
                pdata.interests.add(tag_obj)
                pdata.save()
                async_update_recommendations(user_profile)
                # recommend_products(pdata) 
        return render(request, 'user/search_results.html', {'products': filtered_products, 'query': query})

    # Last step: general sentiment-based search
    sentiment = get_sentiment_score(query)
    print(f"sentiment: {sentiment}")
    if sentiment<0.95:
        return render(request, 'user/search_results.html', {'message': 'No books found.', 'query': query})

    cleaned_query = preprocess_text(query)
    product_descriptions = [f"{p.name} {p.pdes} {p.category.cname}" for p in products]
    cleaned_descriptions = [preprocess_text(desc) for desc in product_descriptions]

    print(f"cleaned _query:{cleaned_query}")
    # vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    # tfidf_matrix = vectorizer.fit_transform(cleaned_descriptions)
    # query_tfidf = vectorizer.transform([cleaned_query])
    # similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    matched_products = []
    query_words = set(cleaned_query.split())

    for candidate_product in products:
        candidate_title=candidate_product.name.lower()
        candidate_desc = candidate_product.pdes.lower()
        candidate_cat = candidate_product.category.cname.lower()
        candidate_sentiment = get_sentiment_score(candidate_desc)

        sim_score = get_semantic_similarity(cleaned_query, candidate_desc)
        sim_score2 = get_semantic_similarity(candidate_cat, cleaned_query)

        # Create set of words from cleaned product description + category
        desc_text = f"{candidate_product.name} {candidate_desc} {candidate_cat}"
        desc_words = set(preprocess_text(desc_text).split())

        print(candidate_product.name)
        print(f"sim_score (desc): {sim_score}, sim_score (cat): {sim_score2}")

        # Add product if it meets semantic threshold
        if sim_score > 0.52 or sim_score2 > 0.45:
            matched_products.append((candidate_product, sim_score))
        # Or if there's a word match
        elif query_words & desc_words:
            matched_products.append((candidate_product, sim_score))


    # Sort the matched products by sim_score (highest first)
    matched_products.sort(key=lambda x: x[1], reverse=True)

    # Extract only the product objects
    result_products = [item[0] for item in matched_products]

    if not result_products:
        return render(request, 'user/search_results.html', {'message': 'No relevant results found.', 'query': query})

    # Add sentiment tag to interest
    if request.user.is_authenticated and pdata:
            

            # Check similarity with existing interests
            existing_interests = [tag.name for tag in pdata.interests.all()]
            already_similar = False

            for interest in existing_interests:
                sim_score = get_semantic_similarity(cleaned_query, interest)
                if sim_score > 0.7:  # Threshold to consider it similar
                    already_similar = True
                    break

            # Only add if it's new (not similar) and sentiment is strong
            if not already_similar and sentiment > 0.9:
                tag_obj, _ = InterestTag.objects.get_or_create(name=cleaned_query)
                pdata.interests.add(tag_obj)
                pdata.save()

                async_update_recommendations(user_profile)

                # # You can now call recommendation engine and update session
                # new_recommendations = get_recommendations_api(pdata)
                # request.session['recommended_products'] = [p.id for p in new_recommendations]


    return render(request, 'user/search_results.html', {
        'products': result_products,
        'query': query
    })

def handle_like_dislike(request):
    if request.method == 'POST':
        user_id = request.session.get('userid')
        product_id = request.POST.get('product_id')
        action = request.POST.get('action')

        if user_id and product_id:
            product_instance = get_object_or_404(product, id=product_id)

            if action == 'like':
                product_instance.likes += 1
            elif action == 'dislike':
                product_instance.dislikes += 1

            product_instance.save()
            return JsonResponse({"status": "success", "likes": product_instance.likes, "dislikes": product_instance.dislikes})

    return JsonResponse({"status": "error", "message": "Invalid request"})

class UserBookInteraction(models.Model):
    """Tracks all user interactions with books"""
    user = models.ForeignKey(profile, on_delete=models.CASCADE)
    book = models.ForeignKey(product, on_delete=models.CASCADE)
    interaction_type = models.CharField(
        max_length=20,
        choices=[
            ('view', 'View'),
            ('purchase', 'Purchase'),
            ('like', 'Like'),
            ('dislike', 'Dislike'),
            ('review', 'Review'),
            ('search', 'Search')
        ]
    )
    weight = models.FloatField(default=1.0)  # Strength of interaction
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=['user', 'book']),
            models.Index(fields=['interaction_type']),
        ]

    def __str__(self):
        return f"{self.user.name} {self.interaction_type} {self.book.name}"
    
# views.py
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404
import json

@login_required
@csrf_exempt
def product_interaction(request, product_id, action):
    if request.method == 'POST':
        product = get_object_or_404(product, id=product_id)
        user = request.user

        # Implement like/dislike logic here:
        if action == 'like':
            # Update like for this user-product pair
            pass
        elif action == 'dislike':
            # Update dislike for this user-product pair
            pass
        else:
            return JsonResponse({'success': False, 'error': 'Invalid action'})

        return JsonResponse({'success': True})

@login_required
@csrf_exempt
def product_view(request, product_id):
    if request.method == 'POST':
        product = get_object_or_404(product, id=product_id)
        user = request.user

        # Log view for user-product here

        return JsonResponse({'success': True})

from django.shortcuts import redirect
from django.utils import translation

def set_language(request):
    lang_code = request.GET.get('language', 'en')
    if lang_code in dict(settings.LANGUAGES).keys():
        request.session[translation.LANGUAGE_SESSION_KEY] = lang_code
    return redirect(request.META.get('HTTP_REFERER', '/'))

# views.py

# user/views.py

# views.py

# views.py
import os
import fitz  # PyMuPDF
from django.conf import settings
from django.http import JsonResponse
from deep_translator import GoogleTranslator  

def get_pdf_filename(product_id):
    try:
        prod = product.objects.get(id=product_id)
        if prod.pdf:  # Check if PDF exists
            filename = prod.pdf.name.split('/')[-1]  # Extract filename only
            return filename
        else:
            return None
    except product.DoesNotExist:
        return None

def extract_pdf_page_text(pdf_path, page_number):
    try:
        doc = fitz.open(pdf_path)
        if page_number < 1 or page_number > doc.page_count:
            return ""
        page = doc.load_page(page_number - 1)
        text = page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"PDF read error: {e}")
        return ""

def translate_text(text, dest_lang):
    try:
        translated = GoogleTranslator(source='auto', target=dest_lang).translate(text)
        return translated
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def translate_pdf_pages_api(request):
    product_id = request.GET.get('product_id')
    page_number = int(request.GET.get('page', 1))
    language = request.GET.get('language', 'en')

    if not product_id:
        return JsonResponse({'error': 'Missing product_id'})

    filename = get_pdf_filename(product_id)
    if not filename:
        return JsonResponse({'error': 'PDF not found for product'})

    pdf_relative_path = os.path.join('static', 'pdfs', filename)
    full_pdf_path = os.path.join(settings.BASE_DIR, pdf_relative_path)

    if not os.path.exists(full_pdf_path):
        return JsonResponse({'error': 'PDF file not found on server'})

    original_text = extract_pdf_page_text(full_pdf_path, page_number)
    if not original_text:
        return JsonResponse({'error': 'Page not found or empty'})

    translated_text = translate_text(original_text, language)

    return JsonResponse({'pages': [{'page': page_number, 'text': translated_text}]})


def read_pdf(request, product_id):
    
    product = get_object_or_404(product, id=product_id)
    original_language = product.language  # Assuming this is stored
    return render(request, 'user/viewdetails.html', {
        
        'product': product,
        'original_language': original_language,
    })
