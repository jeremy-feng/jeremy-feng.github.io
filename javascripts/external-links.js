document$.subscribe(function () {
  var origin = window.location.origin
  document.querySelectorAll("a[href]").forEach(function (link) {
    var href = link.getAttribute("href")
    if (!href || !/^https?:\/\//.test(href) || href.indexOf(origin) === 0) return
    link.setAttribute("target", "_blank")
    var rel = link.getAttribute("rel") || ""
    if (rel.indexOf("noopener") === -1) {
      link.setAttribute("rel", (rel + " noopener").trim())
    }
  })
})
