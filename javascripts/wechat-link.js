document.addEventListener('DOMContentLoaded', function () {
  var el = document.querySelector('.md-social__link[title="WeChat"]');
  if (el) {
    el.addEventListener('click', function (e) {
      e.preventDefault();
      e.stopPropagation();
    });
  }
});