import * as React from "react";
import { TextInput, Textarea } from "@mantine/core";
import { ViserInputComponent } from "./common";
import { GuiTextMessage } from "../WebsocketMessages";
import { GuiComponentContext } from "../ControlPanel/GuiComponentContext";
import { ViewerContext } from "../ViewerContext";
import KeyboardControl from "./KeyboardControl";
import SessionHeartbeat from "./SessionHeartbeat";
import CameraFeeds from "./CameraFeeds";
import RemoteControl from "./RemoteControl";

export default function TextInputComponent({
  uuid,
  value,
  props: { hint, label, disabled, visible, multiline },
}: GuiTextMessage) {
  const { setValue } = React.useContext(GuiComponentContext)!;
  const viewer = React.useContext(ViewerContext)!;
  if (!visible) return null;
  if (label === "so101_keyboard") {
    return <KeyboardControl disabled={disabled} resetKey={hint ?? "0"}
      onChange={(next) => setValue(uuid, next)}/>;
  }
  if (label === "so101_session") {
    // setValue shares one throttled sender that keeps only the latest message, so heartbeats
    // and commands would silently replace each other: lost clicks and false operator timeouts.
    return <SessionHeartbeat onChange={(next) => viewer.mutable.current.sendMessage({
      type: "GuiUpdateMessage", uuid, updates: { value: next },
    })}/>;
  }
  if (label === "so101_cameras") {
    return <CameraFeeds/>;
  }
  if (label === "so101_remote") {
    return <RemoteControl hint={hint ?? ""} onChange={(next) => setValue(uuid, next)}/>;
  }
  return (
    <ViserInputComponent {...{ uuid, hint, label }}>
      {multiline ? (
        <Textarea
          value={value}
          size="xs"
          onChange={(value) => {
            setValue(uuid, value.target.value);
          }}
          styles={{
            input: {
              padding: "0 0.5em",
            },
          }}
          disabled={disabled}
          minRows={2}
          maxRows={6}
          autosize
          resize="vertical"
        />
      ) : (
        <TextInput
          value={value}
          size="xs"
          onChange={(value) => {
            setValue(uuid, value.target.value);
          }}
          styles={{
            input: {
              minHeight: "1.625rem",
              height: "1.625rem",
              padding: "0 0.5em",
            },
          }}
          disabled={disabled}
        />
      )}
    </ViserInputComponent>
  );
}
