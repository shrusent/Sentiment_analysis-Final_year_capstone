from django.http import HttpResponse
from django.template import loader
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.views.decorators.csrf import csrf_exempt
import base64

def index(request):
    template = loader.get_template('index.html')
    return HttpResponse(template.render())


def about(request):
    template = loader.get_template('about.html')
    return HttpResponse(template.render())


def blog(request):
    template = loader.get_template('blog.html')
    return HttpResponse(template.render())


def contact(request):
    template = loader.get_template('contact.html')
    return HttpResponse(template.render())


def finance(request):
    template = loader.get_template('Finance.html')
    return HttpResponse(template.render())


def restaurant_review(request):
    template = loader.get_template('Restaurant review.html')
    return HttpResponse(template.render())


def services(request):
    template = loader.get_template('services.html')
    return HttpResponse(template.render())


def restaurant_review_search(request):
    template = loader.get_template('Restaurant review search.html')
    return HttpResponse(template.render())


@csrf_exempt
def run(request):
    s = request.POST['search_key']
    from ai_models import modela
    res = modela.model1(s)
    context = {
        'r':res
    }
    template = loader.get_template('restaurant review result.html')
    return HttpResponse(template.render(context,request))


@api_view(['POST','GET'])
def res_rev_search(request):
    q = request.data['q']
    #q = "bad food"
    from ai_models import modela
    res = modela.model1(q)
    s = {
        "res":res
    }
    return Response(s)

@api_view(['POST','GET'])
def fin_search(request):

    from ai_models import model_twitter
    res = model_twitter.fn_main()

    r={
        "image":base64.b64encode(res).decode("utf8")
    }
    return Response(r)




