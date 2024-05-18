document.addEventListener("DOMContentLoaded", function () {
  const dropdowns = document.querySelectorAll(".relative");

  dropdowns.forEach((dropdown) => {
    const toggleMenu = dropdown.querySelector("a");
    const chevron = toggleMenu.querySelector("svg");
    toggleMenu.addEventListener("click", function (event) {
      event.preventDefault();
      const submenu = dropdown.querySelector("ul");
      submenu.classList.toggle("hidden");
      chevron.classList.toggle("rotate-180");
    });
  });

  document.addEventListener("click", function (e) {
    dropdowns.forEach((dropdown) => {
      if (!dropdown.contains(e.target)) {
        const submenu = dropdown.querySelector("ul");
        const chevron = dropdown.querySelector("svg");
        submenu.classList.add("hidden");
        chevron.classList.remove("rotate-180");
      }
    });
  });
});
