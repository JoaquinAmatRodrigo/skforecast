function flush() {
  var id = location.hash.replace(/\./g, "\\.")
  if (id.length) {
    console.log(id);
	  // $("" + id).css("background-color","#9f9");
  }
}

$(flush);

// $("html").click(flush)
