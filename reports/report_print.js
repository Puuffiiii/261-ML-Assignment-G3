<script>
document.addEventListener("DOMContentLoaded", function () {
  const toc = document.getElementById("TOC");
  const introduction = document.getElementById("giới-thiệu");
  if (toc && introduction && introduction.parentNode) {
    introduction.parentNode.insertBefore(toc, introduction);
  }
});
</script>
