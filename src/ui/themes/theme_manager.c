#include "theme_manager.h"
#include <glib/gi18n.h>

// Global settings
static BLOUeditThemeType current_theme = BLOUEDIT_THEME_SYSTEM;
static GSettings *settings = NULL;
static GtkCssProvider *theme_provider = NULL;
static GtkCssProvider *custom_provider = NULL;

// Theme CSS
static const char *light_theme_css = 
"@define-color accent_color #3584e4;\n"
"@define-color accent_bg_color @accent_color;\n"
"@define-color accent_fg_color white;\n"
"@define-color destructive_color #e01b24;\n"
"@define-color destructive_bg_color @destructive_color;\n"
"@define-color destructive_fg_color white;\n"
"@define-color success_color #26a269;\n"
"@define-color success_bg_color @success_color;\n"
"@define-color success_fg_color white;\n"
"@define-color warning_color #e5a50a;\n"
"@define-color warning_bg_color @warning_color;\n"
"@define-color warning_fg_color rgba(0, 0, 0, 0.8);\n"
"@define-color error_color #e01b24;\n"
"@define-color error_bg_color @error_color;\n"
"@define-color error_fg_color white;\n"
"@define-color window_bg_color #fafafa;\n"
"@define-color window_fg_color rgba(0, 0, 0, 0.8);\n"
"@define-color view_bg_color #ffffff;\n"
"@define-color view_fg_color @window_fg_color;\n"
"@define-color headerbar_bg_color #ebebeb;\n"
"@define-color headerbar_fg_color @window_fg_color;\n"
"@define-color headerbar_border_color rgba(0, 0, 0, 0.1);\n"
"@define-color headerbar_shade_color rgba(0, 0, 0, 0.07);\n"
"@define-color card_bg_color @view_bg_color;\n"
"@define-color card_fg_color @view_fg_color;\n"
"@define-color card_shade_color rgba(0, 0, 0, 0.07);\n"
"@define-color popover_bg_color @view_bg_color;\n"
"@define-color popover_fg_color @view_fg_color;\n"
"@define-color shade_color rgba(0, 0, 0, 0.07);\n"
"@define-color scrollbar_outline_color rgba(0, 0, 0, 0.1);\n"
"\n"
"/* BLOUedit custom styling */\n"
".blouedit-timeline {\n"
"    background-color: #e0e0e0;\n"
"    color: rgba(0, 0, 0, 0.8);\n"
"}\n"
"\n"
".blouedit-timeline-track {\n"
"    background-color: #f0f0f0;\n"
"    border-bottom: 1px solid rgba(0, 0, 0, 0.1);\n"
"}\n"
"\n"
".blouedit-clip {\n"
"    background-color: #3584e4;\n"
"    color: white;\n"
"    border-radius: 3px;\n"
"}\n"
"\n"
".blouedit-audio-clip {\n"
"    background-color: #26a269;\n"
"    color: white;\n"
"    border-radius: 3px;\n"
"}\n"
"\n"
".blouedit-video-area {\n"
"    background-color: #1e1e1e;\n"
"    color: white;\n"
"    border: 1px solid rgba(0, 0, 0, 0.2);\n"
"}\n"
"\n"
".blouedit-toolbar {\n"
"    background-color: @headerbar_bg_color;\n"
"    border-bottom: 1px solid @headerbar_border_color;\n"
"}\n";

static const char *dark_theme_css = 
"@define-color accent_color #3584e4;\n"
"@define-color accent_bg_color @accent_color;\n"
"@define-color accent_fg_color white;\n"
"@define-color destructive_color #e01b24;\n"
"@define-color destructive_bg_color @destructive_color;\n"
"@define-color destructive_fg_color white;\n"
"@define-color success_color #26a269;\n"
"@define-color success_bg_color @success_color;\n"
"@define-color success_fg_color white;\n"
"@define-color warning_color #e5a50a;\n"
"@define-color warning_bg_color @warning_color;\n"
"@define-color warning_fg_color rgba(0, 0, 0, 0.8);\n"
"@define-color error_color #e01b24;\n"
"@define-color error_bg_color @error_color;\n"
"@define-color error_fg_color white;\n"
"@define-color window_bg_color #242424;\n"
"@define-color window_fg_color rgba(255, 255, 255, 0.9);\n"
"@define-color view_bg_color #1e1e1e;\n"
"@define-color view_fg_color rgba(255, 255, 255, 0.9);\n"
"@define-color headerbar_bg_color #303030;\n"
"@define-color headerbar_fg_color rgba(255, 255, 255, 0.9);\n"
"@define-color headerbar_border_color rgba(0, 0, 0, 0.8);\n"
"@define-color headerbar_shade_color rgba(0, 0, 0, 0.36);\n"
"@define-color card_bg_color @view_bg_color;\n"
"@define-color card_fg_color @view_fg_color;\n"
"@define-color card_shade_color rgba(0, 0, 0, 0.36);\n"
"@define-color popover_bg_color @view_bg_color;\n"
"@define-color popover_fg_color @view_fg_color;\n"
"@define-color shade_color rgba(0, 0, 0, 0.36);\n"
"@define-color scrollbar_outline_color rgba(0, 0, 0, 0.5);\n"
"\n"
"/* BLOUedit custom styling */\n"
".blouedit-timeline {\n"
"    background-color: #2d2d2d;\n"
"    color: rgba(255, 255, 255, 0.9);\n"
"}\n"
"\n"
".blouedit-timeline-track {\n"
"    background-color: #363636;\n"
"    border-bottom: 1px solid rgba(0, 0, 0, 0.8);\n"
"}\n"
"\n"
".blouedit-clip {\n"
"    background-color: #3584e4;\n"
"    color: white;\n"
"    border-radius: 3px;\n"
"}\n"
"\n"
".blouedit-audio-clip {\n"
"    background-color: #26a269;\n"
"    color: white;\n"
"    border-radius: 3px;\n"
"}\n"
"\n"
".blouedit-video-area {\n"
"    background-color: #121212;\n"
"    color: white;\n"
"    border: 1px solid rgba(255, 255, 255, 0.1);\n"
"}\n"
"\n"
".blouedit-toolbar {\n"
"    background-color: @headerbar_bg_color;\n"
"    border-bottom: 1px solid @headerbar_border_color;\n"
"}\n";

static void on_dark_preference_changed(GtkSettings *settings,
                                      GParamSpec *pspec,
                                      gpointer user_data) {
    if (current_theme == BLOUEDIT_THEME_SYSTEM) {
        gboolean prefer_dark_theme;
        g_object_get(settings, "gtk-application-prefer-dark-theme", 
                    &prefer_dark_theme, NULL);
        
        if (prefer_dark_theme) {
            gtk_css_provider_load_from_data(theme_provider, dark_theme_css, -1);
        } else {
            gtk_css_provider_load_from_data(theme_provider, light_theme_css, -1);
        }
    }
}

static void apply_theme() {
    g_autoptr(GtkSettings) gtk_settings = gtk_settings_get_default();
    
    switch (current_theme) {
        case BLOUEDIT_THEME_LIGHT:
            g_object_set(gtk_settings, "gtk-application-prefer-dark-theme", FALSE, NULL);
            gtk_css_provider_load_from_data(theme_provider, light_theme_css, -1);
            break;
            
        case BLOUEDIT_THEME_DARK:
            g_object_set(gtk_settings, "gtk-application-prefer-dark-theme", TRUE, NULL);
            gtk_css_provider_load_from_data(theme_provider, dark_theme_css, -1);
            break;
            
        case BLOUEDIT_THEME_SYSTEM:
            // Follow system preference
            gboolean prefer_dark_theme;
            g_object_get(gtk_settings, "gtk-application-prefer-dark-theme", 
                        &prefer_dark_theme, NULL);
            
            if (prefer_dark_theme) {
                gtk_css_provider_load_from_data(theme_provider, dark_theme_css, -1);
            } else {
                gtk_css_provider_load_from_data(theme_provider, light_theme_css, -1);
            }
            break;
    }
    
    // Save the setting
    if (settings) {
        g_settings_set_enum(settings, "color-scheme", current_theme);
    }
}

static void on_theme_changed(GtkCheckButton *button, gpointer user_data) {
    if (!gtk_check_button_get_active(button)) return;
    
    const char *theme_name = g_object_get_data(G_OBJECT(button), "theme-name");
    
    if (g_strcmp0(theme_name, "light") == 0) {
        blouedit_theme_manager_set_theme(BLOUEDIT_THEME_LIGHT);
    } else if (g_strcmp0(theme_name, "dark") == 0) {
        blouedit_theme_manager_set_theme(BLOUEDIT_THEME_DARK);
    } else if (g_strcmp0(theme_name, "system") == 0) {
        blouedit_theme_manager_set_theme(BLOUEDIT_THEME_SYSTEM);
    }
}

static void on_color_selected(GtkColorButton *button, gpointer user_data) {
    const char *color_type = g_object_get_data(G_OBJECT(button), "color-type");
    GdkRGBA rgba;
    char *color_string;
    
    gtk_color_chooser_get_rgba(GTK_COLOR_CHOOSER(button), &rgba);
    color_string = gdk_rgba_to_string(&rgba);
    
    if (settings) {
        char *key = g_strdup_printf("%s-color", color_type);
        g_settings_set_string(settings, key, color_string);
        g_free(key);
    }
    
    // Update the theme with all current custom colors
    blouedit_theme_manager_set_custom_colors(
        g_settings_get_string(settings, "primary-color"),
        g_settings_get_string(settings, "accent-color"),
        g_settings_get_string(settings, "background-color"),
        g_settings_get_string(settings, "text-color")
    );
    
    g_free(color_string);
}

void blouedit_theme_manager_init(GtkApplication *application) {
    // Create CSS providers
    theme_provider = gtk_css_provider_new();
    custom_provider = gtk_css_provider_new();
    
    // Add the providers to the default screen
    GdkDisplay *display = gdk_display_get_default();
    gtk_style_context_add_provider_for_display(display,
                                              GTK_STYLE_PROVIDER(theme_provider),
                                              GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);
    
    gtk_style_context_add_provider_for_display(display,
                                              GTK_STYLE_PROVIDER(custom_provider),
                                              GTK_STYLE_PROVIDER_PRIORITY_APPLICATION + 1);
    
    // Connect to the dark theme preference change
    g_autoptr(GtkSettings) gtk_settings = gtk_settings_get_default();
    g_signal_connect(gtk_settings, "notify::gtk-application-prefer-dark-theme",
                    G_CALLBACK(on_dark_preference_changed), NULL);
    
    // Initialize GSettings
    settings = g_settings_new("com.blouedit.BLOUedit");
    
    // Load the theme preference from settings
    current_theme = g_settings_get_enum(settings, "color-scheme");
    
    // Apply the theme
    apply_theme();
    
    // Load custom colors
    blouedit_theme_manager_set_custom_colors(
        g_settings_get_string(settings, "primary-color"),
        g_settings_get_string(settings, "accent-color"),
        g_settings_get_string(settings, "background-color"),
        g_settings_get_string(settings, "text-color")
    );
}

void blouedit_theme_manager_set_theme(BLOUeditThemeType theme_type) {
    current_theme = theme_type;
    apply_theme();
}

BLOUeditThemeType blouedit_theme_manager_get_theme(void) {
    return current_theme;
}

void blouedit_theme_manager_set_custom_colors(const char *primary_color, 
                                            const char *accent_color,
                                            const char *background_color,
                                            const char *text_color) {
    char *css = g_strdup_printf(
        ":root {\n"
        "  --primary-color: %s;\n"
        "  --accent-color: %s;\n"
        "  --background-color: %s;\n"
        "  --text-color: %s;\n"
        "}\n"
        "\n"
        ".blouedit-clip {\n"
        "  background-color: var(--primary-color);\n"
        "}\n"
        "\n"
        ".blouedit-audio-clip {\n"
        "  background-color: var(--accent-color);\n"
        "}\n",
        primary_color ? primary_color : "#3584e4",
        accent_color ? accent_color : "#26a269",
        background_color ? background_color : "inherit",
        text_color ? text_color : "inherit"
    );
    
    gtk_css_provider_load_from_data(custom_provider, css, -1);
    g_free(css);
}

void blouedit_theme_manager_apply_css(const char *css) {
    gtk_css_provider_load_from_data(custom_provider, css, -1);
}

GtkWidget* blouedit_theme_manager_create_settings_widget(void) {
    GtkWidget *box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 12);
    gtk_widget_set_margin_start(box, 24);
    gtk_widget_set_margin_end(box, 24);
    gtk_widget_set_margin_top(box, 24);
    gtk_widget_set_margin_bottom(box, 24);
    
    // Theme section
    GtkWidget *theme_label = gtk_label_new(NULL);
    gtk_label_set_markup(GTK_LABEL(theme_label), "<b>Theme</b>");
    gtk_widget_set_halign(theme_label, GTK_ALIGN_START);
    gtk_box_append(GTK_BOX(box), theme_label);
    
    // Theme options
    GtkWidget *theme_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 12);
    gtk_widget_set_margin_start(theme_box, 12);
    gtk_widget_set_margin_top(theme_box, 6);
    
    // Light theme option
    GtkWidget *light_button = gtk_check_button_new_with_label(_("Light"));
    g_object_set_data(G_OBJECT(light_button), "theme-name", "light");
    
    // Dark theme option
    GtkWidget *dark_button = gtk_check_button_new_with_label(_("Dark"));
    g_object_set_data(G_OBJECT(dark_button), "theme-name", "dark");
    gtk_check_button_set_group(GTK_CHECK_BUTTON(dark_button), GTK_CHECK_BUTTON(light_button));
    
    // System theme option
    GtkWidget *system_button = gtk_check_button_new_with_label(_("System"));
    g_object_set_data(G_OBJECT(system_button), "theme-name", "system");
    gtk_check_button_set_group(GTK_CHECK_BUTTON(system_button), GTK_CHECK_BUTTON(light_button));
    
    // Set the active button
    switch (current_theme) {
        case BLOUEDIT_THEME_LIGHT:
            gtk_check_button_set_active(GTK_CHECK_BUTTON(light_button), TRUE);
            break;
        case BLOUEDIT_THEME_DARK:
            gtk_check_button_set_active(GTK_CHECK_BUTTON(dark_button), TRUE);
            break;
        case BLOUEDIT_THEME_SYSTEM:
            gtk_check_button_set_active(GTK_CHECK_BUTTON(system_button), TRUE);
            break;
    }
    
    // Connect signals
    g_signal_connect(light_button, "toggled", G_CALLBACK(on_theme_changed), NULL);
    g_signal_connect(dark_button, "toggled", G_CALLBACK(on_theme_changed), NULL);
    g_signal_connect(system_button, "toggled", G_CALLBACK(on_theme_changed), NULL);
    
    gtk_box_append(GTK_BOX(theme_box), light_button);
    gtk_box_append(GTK_BOX(theme_box), dark_button);
    gtk_box_append(GTK_BOX(theme_box), system_button);
    
    gtk_box_append(GTK_BOX(box), theme_box);
    
    // Separator
    GtkWidget *separator = gtk_separator_new(GTK_ORIENTATION_HORIZONTAL);
    gtk_widget_set_margin_top(separator, 12);
    gtk_widget_set_margin_bottom(separator, 12);
    gtk_box_append(GTK_BOX(box), separator);
    
    // Custom colors section
    GtkWidget *colors_label = gtk_label_new(NULL);
    gtk_label_set_markup(GTK_LABEL(colors_label), "<b>Custom Colors</b>");
    gtk_widget_set_halign(colors_label, GTK_ALIGN_START);
    gtk_box_append(GTK_BOX(box), colors_label);
    
    // Color picker grid
    GtkWidget *color_grid = gtk_grid_new();
    gtk_grid_set_row_spacing(GTK_GRID(color_grid), 6);
    gtk_grid_set_column_spacing(GTK_GRID(color_grid), 12);
    gtk_widget_set_margin_start(color_grid, 12);
    gtk_widget_set_margin_top(color_grid, 6);
    
    // Primary color
    GtkWidget *primary_label = gtk_label_new(_("Primary:"));
    gtk_widget_set_halign(primary_label, GTK_ALIGN_END);
    
    GtkWidget *primary_button = gtk_color_button_new();
    g_object_set_data(G_OBJECT(primary_button), "color-type", "primary");
    
    GdkRGBA primary_rgba;
    gdk_rgba_parse(&primary_rgba, g_settings_get_string(settings, "primary-color"));
    gtk_color_chooser_set_rgba(GTK_COLOR_CHOOSER(primary_button), &primary_rgba);
    
    g_signal_connect(primary_button, "color-set", G_CALLBACK(on_color_selected), NULL);
    
    gtk_grid_attach(GTK_GRID(color_grid), primary_label, 0, 0, 1, 1);
    gtk_grid_attach(GTK_GRID(color_grid), primary_button, 1, 0, 1, 1);
    
    // Accent color
    GtkWidget *accent_label = gtk_label_new(_("Accent:"));
    gtk_widget_set_halign(accent_label, GTK_ALIGN_END);
    
    GtkWidget *accent_button = gtk_color_button_new();
    g_object_set_data(G_OBJECT(accent_button), "color-type", "accent");
    
    GdkRGBA accent_rgba;
    gdk_rgba_parse(&accent_rgba, g_settings_get_string(settings, "accent-color"));
    gtk_color_chooser_set_rgba(GTK_COLOR_CHOOSER(accent_button), &accent_rgba);
    
    g_signal_connect(accent_button, "color-set", G_CALLBACK(on_color_selected), NULL);
    
    gtk_grid_attach(GTK_GRID(color_grid), accent_label, 0, 1, 1, 1);
    gtk_grid_attach(GTK_GRID(color_grid), accent_button, 1, 1, 1, 1);
    
    // Background color
    GtkWidget *bg_label = gtk_label_new(_("Background:"));
    gtk_widget_set_halign(bg_label, GTK_ALIGN_END);
    
    GtkWidget *bg_button = gtk_color_button_new();
    g_object_set_data(G_OBJECT(bg_button), "color-type", "background");
    
    GdkRGBA bg_rgba;
    gdk_rgba_parse(&bg_rgba, g_settings_get_string(settings, "background-color"));
    gtk_color_chooser_set_rgba(GTK_COLOR_CHOOSER(bg_button), &bg_rgba);
    
    g_signal_connect(bg_button, "color-set", G_CALLBACK(on_color_selected), NULL);
    
    gtk_grid_attach(GTK_GRID(color_grid), bg_label, 0, 2, 1, 1);
    gtk_grid_attach(GTK_GRID(color_grid), bg_button, 1, 2, 1, 1);
    
    // Text color
    GtkWidget *text_label = gtk_label_new(_("Text:"));
    gtk_widget_set_halign(text_label, GTK_ALIGN_END);
    
    GtkWidget *text_button = gtk_color_button_new();
    g_object_set_data(G_OBJECT(text_button), "color-type", "text");
    
    GdkRGBA text_rgba;
    gdk_rgba_parse(&text_rgba, g_settings_get_string(settings, "text-color"));
    gtk_color_chooser_set_rgba(GTK_COLOR_CHOOSER(text_button), &text_rgba);
    
    g_signal_connect(text_button, "color-set", G_CALLBACK(on_color_selected), NULL);
    
    gtk_grid_attach(GTK_GRID(color_grid), text_label, 0, 3, 1, 1);
    gtk_grid_attach(GTK_GRID(color_grid), text_button, 1, 3, 1, 1);
    
    gtk_box_append(GTK_BOX(box), color_grid);
    
    // Reset button
    GtkWidget *reset_button = gtk_button_new_with_label(_("Reset to Defaults"));
    gtk_widget_set_margin_top(reset_button, 12);
    gtk_widget_set_halign(reset_button, GTK_ALIGN_END);
    
    gtk_box_append(GTK_BOX(box), reset_button);
    
    return box;
} 