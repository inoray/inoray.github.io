{% extends 'markdown.tpl' %}

{%- block header -%}
---
title: "{{resources['metadata']['name']}}"
excerpt: ""
header:
    overlay_image: /assets/images/header_image.png
    overlay_filter: 0.5
    caption: ""
categories:
tags:
    - python
    - notebook
toc: true
---
{%- endblock header -%}

{% block in_prompt %}
**In [{{ cell.execution_count }}]:**
{% endblock in_prompt %}

{% block input %}
{{ '{% highlight python linenos %}' }}
{{ cell.source }}
{{ '{% endhighlight %}' }}
{% endblock input %}

{% block markdowncell scoped %} 
{{ cell.source | wrap_text(80) }} 
{% endblock markdowncell %} 

{% block headingcell scoped %}
{{ '#' * cell.level }} {{ cell.source | replace('\n', ' ') }}
{% endblock headingcell %}
