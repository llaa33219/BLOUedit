#include "application.h"

#include <glib/gi18n.h>
#include <locale.h>

int
main (int   argc,
      char *argv[])
{
  auto app = blouedit::Application::create();
  return app->run(argc, argv);
} 