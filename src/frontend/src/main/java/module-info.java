module org.frontend {
    requires javafx.controls;
    requires javafx.fxml;
    requires java.net.http;
    requires org.json;


    opens org.frontend to javafx.fxml;
    exports org.frontend;
}